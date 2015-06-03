require 'dp'

--[[command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on BillionWords dataset using a Neural Network and SoftmaxTree')
cmd:text('The network contains 3 or more layers. An input dictionary, one ore many dense hidden layer, and a fully connected output layer')
cmd:text('Example:')
cmd:text('$> th languagemodel.lua --small --batchSize 512 ')
cmd:text('$> th languagemodel.lua --tiny --batchSize 512 ')
cmd:text('$> th languagemodel.lua --tiny --batchSize 512 --accUpdate --validEpochSize 10000 --trainEpochSize 100000 --softmaxtree')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--schedule', '{[250]=0.01, [350]=0.001}', 'learning rate schedule')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--batchSize', 256, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate updates inplace using accUpdateGradParameters')
cmd:option('--contextSize', 5, 'number of words preceding the next word used to predict the target word')
cmd:option('--inputEmbeddingSize', 100, 'number of neurons per word embedding')
cmd:option('--hiddenSize', '{200}', 'number of hidden units used for hidden layer')
cmd:option('--outputEmbeddingSize', 100, 'number of hidden units at softmaxtree')
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('--softmaxforest', false, 'use SoftmaxForest instead of SoftmaxTree (uses more memory)')
cmd:option('--forestGaterSize', '{}', 'size of hidden layers used for forest gater (trees are experts)')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dropout', false, 'use dropout on hidden units')
cmd:option('--small', false, 'use a small (1/30th) subset of the training set')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set')
cmd:option('--trainEpochSize', 1000000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 100000, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
opt.hiddenSize = dp.returnString(opt.hiddenSize)
opt.forestGaterSize = dp.returnString(opt.forestGaterSize)
opt.schedule = dp.returnString(opt.schedule)
if not opt.silent then
   table.print(opt)
end


--[[data]]--

local train_file = 'train_data.th7' 
if opt.small then 
   train_file = 'train_small.th7'
elseif opt.tiny then 
   train_file = 'train_tiny.th7'
end

local ds = dp.BillionWords{
   context_size = opt.contextSize, train_file = train_file
}

--[[Model]]--

-- neural network language model
nnlm = nn.Sequential()

-- input layer
-- lookuptable that contains the word embeddings that will be learned
nnlm:extend(
   nn.Dictionary(ds:vocabularySize(), opt.inputEmbeddingSize, opt.accUpdate),
   nn.Collapse(2)
)

dp.vprint(not opt.silent, "Input to first hidden layer has "..
   opt.contextSize*opt.inputEmbeddingSize.." neurons.")

-- hidden layer(s)
inputSize = opt.contextSize*opt.inputEmbeddingSize
opt.hiddenSize[#opt.hiddenSize + 1] = opt.outputEmbeddingSize
for i,hiddenSize in ipairs(opt.hiddenSize) do
   if opt.dropout then
      nnlm:add(nn.Dropout())
   end
   nnlm:add(nn.Linear(inputSize, hiddenSize))
   if opt.batchNorm then
      nnlm:add(nn.BatchNormalization(hiddenSize))
   end
   nnlm:add(nn.Tanh())
   inputSize = hiddenSize
end

-- output layer
if opt.dropout then
   nnlm:add(nn.Dropout())
end
if opt.softmaxforest or opt.softmaxtree then
   -- input to nnlm is {inputs, targets} for nn.SoftMaxTree
   local para = nn.ParallelTable()
   para:add(nnlm):add(opt.cuda and nn.Convert() or nn.Identity()) 
   nnlm = nn.Sequential()
   nnlm:add(para)
   if opt.softmaxforest then -- requires a lot more memory
      local trees = {ds:hierarchy('word_tree1.th7'), ds:hierarchy('word_tree2.th7'), ds:hierarchy('word_tree3.th7')}
      local rootIds = {880542,880542,880542}
      nnlm:add(nn.SoftMaxForest(inputSize, trees, rootIds, opt.forestGaterSize, nn.Tanh(), opt.accUpdate))
      opt.softmaxtree = true
   elseif opt.softmaxtree then
      local tree, root = ds:frequencyTree()
      nnlm:add(nn.SoftMaxTree(inputSize, tree, root, opt.accUpdate))
   end
else
   print("Warning: you are using full LogSoftMax for last layer, which "..
      "is really slow (800,000 x outputEmbeddingSize multiply adds "..
      "per example. Try --softmaxtree instead.")
   nnlm:add(nn.Linear(inputSize, ds:vocabularySize()))
   nnlm:add(nn.LogSoftMax())
end

--[[Propagators]]--

train = dp.Optimizer{
   loss = opt.softmaxtree and nn.TreeNLLCriterion() or nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
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
   valid = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{
         epoch_size = opt.validEpochSize, batch_size = opt.batchSize
      },
      progress = opt.progress
   }
   tester = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }
end

--[[Experiment]]--

xp = dp.Experiment{
   model = nnlm,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries, 
         error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','perplexity','ppl'}
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}
if opt.softmaxtree then
   -- makes it forward {input, target} instead of just input
   xp:includeTarget()
end

--[[GPU or CPU]]--

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   if opt.softmaxtree then
      require 'cunnx'
   end
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Model :"
   print(nnlm)
end

xp:run(ds)
