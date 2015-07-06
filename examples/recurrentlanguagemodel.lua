require 'dp'
require 'rnn'

version = 5

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
cmd:option('--adaptiveDecay', false, 'use adaptive learning rate')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.1, 'factor by which learning rate is decayed.')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate updates inplace using accUpdateGradParameters')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--xpPath', '', 'path to a previously saved model')

--[[ recurrent layer ]]--
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 200, 'number of hidden units used in Simple RNN')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout on hidden neurons (not recommended)')

--[[ output layer ]]--
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('--softmaxforest', false, 'use SoftmaxForest instead of SoftmaxTree (uses more memory)')
cmd:option('--forestGaterSize', '{}', 'size of hidden layers used for forest gater (trees are experts)') 

--[[ data ]]--
cmd:option('--dataset', 'BillionWords', 'which dataset to use : BillionWords | PennTreeBank | TextSource')
cmd:option('--trainEpochSize', 400000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 24000, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--dataPath', dp.DATA_DIR,  'path to data directory')
-- BillionWords
cmd:option('--small', false, 'use a small (1/30th) subset of the training set (BillionWors only)')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set (BillionWors only)')
-- TextSource
cmd:option('--trainFile', 'train.txt', 'filename containing tokenized training text data')
cmd:option('--validFile', 'valid.txt', 'filename containing tokenized validation text data')
cmd:option('--testFile', 'test.txt', 'filename containing tokenized test text data')

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

if opt.dataset == 'BillionWords' then
   ds = dp.BillionWords{
      train_file=train_file, load_all=false, 
      context_size=opt.rho, recurrent=true
   }
   ds:loadTrain()
   if not opt.trainOnly then
      ds:loadValid()
      ds:loadTest()
   end
elseif opt.dataset == 'PennTreeBank' then
   ds = dp.PennTreeBank{context_size=opt.rho,recurrent=true}
   ds:testSet():contextSize(1) -- so that it works with dp.Sampler
   ds:validSet():contextSize(1)
   assert(not opt.softmaxforest, "SoftMaxForest not supported with PennTreeBank")
elseif opt.dataset == 'TextSource' then
   ds = dp.TextSource{
      context_size=opt.rho, recurrent=true,
      name='rnnlm', data_path = opt.dataPath,
      train=opt.trainFile, valid=opt.validFile, test=opt.testFile
   }
   ds:testSet():contextSize(1) -- so that it works with dp.Sampler
   ds:validSet():contextSize(1)
   assert(not opt.softmaxforest, "SoftMaxForest not supported with TextSource")
else
   error"Unrecognized --dataset"
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
   else
      xp:float()
   end
   xp:run(ds)
   os.exit()
end

--[[Model]]--

-- language model
lm = nn.Sequential()
lm:add(nn.DontCast(nn.SplitTable(1,1):dontBackward():type('torch.IntTensor'))) -- tensor to table of tensors

-- simple recurrent neural network
rnn = nn.Recurrent(
   opt.hiddenSize, -- first step will use nn.Add
   nn.Dictionary(ds:vocabularySize(), opt.hiddenSize, opt.accUpdate), -- input layer is a lookup table
   nn.Linear(opt.hiddenSize, opt.hiddenSize), -- feedback layer (recurrence)
   nn.Sigmoid(), -- transfer function 
   99999 -- maximum number of time-steps per sequence
)
if opt.zeroFirst then
   -- this is equivalent to forwarding a zero vector through the feedback layer
   rnn.startModule:share(rnn.feedbackModule, 'bias')
end
seq = nn.Sequencer(rnn)
if not opt.dataset == 'BillionWords' then
   -- evaluation will recurse a single continuous sequence
   seq:remember()
end
lm:add(seq)

-- output layer
if opt.softmaxforest or opt.softmaxtree then
   -- input to nnlm is {inputs, targets} for nn.SoftMaxTree
   local para = nn.ParallelTable()
   para:add(lm):add(nn.Sequencer(nn.Convert())) 
   lm = nn.Sequential()
   lm:add(para)
   lm:add(nn.ZipTable())
   if opt.softmaxforest then -- requires a lot more memory
      local trees = {ds:hierarchy('word_tree1.th7'), ds:hierarchy('word_tree2.th7'), ds:hierarchy('word_tree3.th7')}
      local rootIds = {880542,880542,880542}
      softmax = nn.SoftMaxForest(opt.hiddenSize, trees, rootIds, opt.forestGaterSize, nn.Tanh(), opt.accUpdate)
      opt.softmaxtree = true
   elseif opt.softmaxtree then -- uses frequency based tree
      local tree, root = ds:frequencyTree()
      softmax = nn.SoftMaxTree(opt.hiddenSize, tree, root, opt.accUpdate)
   end
else
   if #ds:vocabulary() > 50000 then
      print("Warning: you are using full LogSoftMax for last layer, which "..
         "is really slow (800,000 x outputEmbeddingSize multiply adds "..
         "per example. Try --softmaxtree instead.")
   end
   softmax = nn.Sequential()
   softmax:add(nn.Linear(opt.hiddenSize, ds:vocabularySize()))
   softmax:add(nn.LogSoftMax())
end
lm:add(nn.Sequencer(softmax))

--[[Propagators]]--
if opt.adaptiveDecay then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
end
opt.lastEpoch = 0
train = dp.Optimizer{
   loss = nn.SequencerCriterion(
      opt.softmaxtree and nn.TreeNLLCriterion() 
         or nn.ModuleCriterion(
            nn.ClassNLLCriterion(), 
            nn.Identity(), 
            opt.cuda and nn.Convert() or nn.Identity()
         )
   ),
   callback = function(model, report) 
      -- learning rate decay
      if opt.lastEpoch < report.epoch and ad and ad.decay ~= 1 or opt.schedule[report.epoch] then
         opt.learningRate = ad and opt.learningRate*ad.decay or opt.schedule[report.epoch]
         if ad then ad.decay = 1 end
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
      opt.lastEpoch = report.epoch
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
   valid = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = torch.isTypeOf(ds, 'dp.TextSource') and dp.Sampler{batch_size = opt.contextSize} 
         or dp.SentenceSampler{epoch_size = opt.validEpochSize, batch_size = 1, max_size = 100},
      progress = opt.progress
   }
   tester = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = torch.isTypeOf(ds, 'dp.TextSource') and dp.Sampler{batch_size = opt.contextSize} 
         or dp.SentenceSampler{batch_size = 1, max_size = 100}  -- Note : remove max_size for exact test set perplexity (will cost more memory)
   }
end

--[[Experiment]]--
xp = dp.Experiment{
   model = lm,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      ad,
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries, 
         error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','perplexity','ppl'}
      }
   },
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

xp:run(ds)
