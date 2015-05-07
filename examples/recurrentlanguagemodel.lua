require 'dp'
require 'rnn'

version = 4

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on BillionWords dataset using a Simple Recurrent Neural Network')
cmd:text('Example:')
cmd:text('$> th recurrentlanguagemodel.lua --small --batchSize 64 ')
cmd:text('$> th recurrentlanguagemodel.lua --tiny --batchSize 64 ')
cmd:text('$> th recurrentlanguagemodel.lua --tiny --batchSize 64 --rho 5 --validEpochSize 10000 --trainEpochSize 100000 --softmaxtree')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--schedule', '{[250]=0.01, [350]=0.001}', 'learning rate schedule')
cmd:option('--momentum', 0, 'momentum')
--cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
--cmd:option('--decayFactor', 0.1, 'factor by which learning rate is decayed.')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate updates inplace using accUpdateGradParameters')

--[[ recurrent layer ]]--
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 200, 'number of hidden units used in Simple RNN')
cmd:option('--dropout', false, 'apply dropout on hidden neurons (not recommended)')

--[[ output layer ]]--
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('--softmaxforest', false, 'use SoftmaxForest instead of SoftmaxTree (uses more memory)')
cmd:option('--forestGaterSize', '{}', 'size of hidden layers used for forest gater (trees are experts)') 

--[[ data ]]--
cmd:option('--small', false, 'use a small (1/30th) subset of the training set')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set')
cmd:option('--trainEpochSize', 1000000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 100000, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
if not opt.silent then
   table.print(opt)
end

if opt.xpPath ~= '' then
   -- check that saved model exists
   assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
end

--[[data]]--
local train_file = 'train_data.th7' 
if opt.small then 
   train_file = 'train_small.th7'
elseif opt.tiny then 
   train_file = 'train_tiny.th7'
end

datasource = dp.BillionWords{
   train_file=train_file, load_all=false, 
   context_size=opt.rho, recurrent=true
}
datasource:loadTrain()
if not opt.trainOnly then
   --datasource:loadValid()
   datasource:loadTest()
end

--[[Saved experiment]]--
if opt.xpPath ~= '' then
   if opt.cuda then
      require 'cunnx'
      cutorch.setDevice(opt.useDevice)
   end
   xp = torch.load(opt.xpPath)
   if opt.cuda then
      xp:cuda()
   end
   xp:run(datasource)
   os.exit()
end

--[[Model]]--

-- language model
lm = nn.Sequential()
lm:add(nn.DontCast(nn.SplitTable(1,1):dontBackward():type('torch.IntTensor'))) -- tensor to table of tensors

-- simple recurrent neural network
rnn = nn.Recurrent(
   opt.hiddenSize, 
   nn.Dictionary(datasource:vocabularySize(), opt.hiddenSize, opt.accUpdate),
   nn.Linear(opt.hiddenSize, opt.hiddenSize), nn.Sigmoid(), 99999
)
lm:add(nn.Sequencer(rnn))

-- output layer
if opt.softmaxforest or opt.softmaxtree then
   -- input to nnlm is {inputs, targets} for nn.SoftMaxTree
   local para = nn.ParallelTable()
   para:add(lm):add(nn.Sequencer(nn.Convert())) 
   lm = nn.Sequential()
   lm:add(para)
   lm:add(nn.ZipTable())
   if opt.softmaxforest then -- requires a lot more memory
      local trees = {datasource:hierarchy('word_tree1.th7'), datasource:hierarchy('word_tree2.th7'), datasource:hierarchy('word_tree3.th7')}
      local rootIds = {880542,880542,880542}
      softmax = nn.SoftMaxForest(opt.hiddenSize, trees, rootIds, opt.forestGaterSize, nn.Tanh(), opt.accUpdate)
      opt.softmaxtree = true
   elseif opt.softmaxtree then
      softmax = nn.SoftMaxTree(opt.hiddenSize, datasource:hierarchy(), 880542, opt.accUpdate)
   end
else
   print("Warning: you are using full LogSoftMax for last layer, which "..
      "is really slow (800,000 x outputEmbeddingSize multiply adds "..
      "per example. Try --softmaxtree instead.")
   softmax = nn.Sequential()
   softmax:add(nn.Linear(opt.hiddenSize, datasource:vocabularySize()))
   softmax:add(nn.LogSoftMax())
end
lm:add(nn.Sequencer(softmax))

--[[Propagators]]--
train = dp.Optimizer{
   loss = nn.SequencerCriterion(opt.softmaxtree and nn.TreeNLLCriterion() or nn.ClassNLLCriterion()),
   callback = function(model, report) 
      opt.learningRate = opt.schedule[report.epoch] or opt.learningRate
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Perplexity(),  
   sampler = dp.RandomSampler{
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   acc_update = opt.accUpdate,
   progress = opt.progress
}

if not opt.trainOnly then
   local batchSize = math.min(math.max(1, opt.batchSize*opt.rho/25), opt.batchSize)
   print("evaluation batchSize", batchSize)
   valid = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = dp.SentenceSampler{
         epoch_size = opt.validEpochSize, batch_size = batchSize
      },
      progress = opt.progress
   }
   tester = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = dp.SentenceSampler{batch_size = batchSize}
   }
end

--[[Experiment]]--
xp = dp.Experiment{
   model = lm,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = (not opt.trainOnly) and {
      dp.FileLogger(),
      dp.EarlyStopper{max_epochs = opt.maxTries}
   } or nil,
   random_seed = os.time(),
   max_epoch = opt.maxEpoch,
   target_module = nn.SplitTable(1,1):type('torch.IntTensor')
}
if opt.softmaxtree then
   -- makes it forward {input, target} instead of just input
   xp:includeTarget()
end

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   if opt.softmaxtree or opt.softmaxforest then
      require 'cunnx'
   end
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Language Model :"
   print(lm)
end

xp:run(datasource)
