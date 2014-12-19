require 'dp'

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
cmd:option('--lrScales', '{1,1}', 'layer-wise learning rate scales : learningRate*lrScale')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.1, 'factor by which learning rate is decayed.')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 1, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--sparseInit', false, 'initialize inputs using a sparse initialization (as opposed to the default normal initialization)')

--[[ recurrent layer ]]--
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--updateInterval', -1, 'BPTT every updateInterval steps (and implicitly, at the end of each sentence). Defaults to --rho')
cmd:option('--forceForget', false, 'force the recurrent layer to forget its past activations after each update')
cmd:option('--hiddenSize', 200, 'number of hidden units used in Simple RNN')
cmd:option('--dropout', false, 'apply dropout on hidden neurons (not recommended)')

--[[ output layer ]]--
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')
--cmd:option('--accUpdate', false, 'accumulate output layer updates inplace. Note that this will cause BPTT instability, but will cost less memory.')

--[[ data ]]--
cmd:option('--small', false, 'use a small (1/30th) subset of the training set')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set')
cmd:option('--trainEpochSize', 1000000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 100000, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--progress', false, 'print progress bar')

version = 3

cmd:option('--xpPath', '', 'path to a previously saved model')

cmd:text()
opt = cmd:parse(arg or {})
opt.updateInterval = opt.updateInterval == -1 and opt.rho or opt.updateInterval
table.print(opt)

opt.lrScales = table.fromString(opt.lrScales)

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

local datasource = dp.BillionWords{train_file = train_file, load_all=false}
datasource:loadTrain()
if not opt.trainOnly then
   datasource:loadValid()
   datasource:loadTest()
end

--[[Saved experiment]]--
if opt.xpPath ~= '' then
   if opt.cuda then
      require 'cutorch'
      require 'cunn'
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

-- build the last layer first:
local softmax
if opt.softmaxtree then
   softmax = dp.SoftmaxTree{
      input_size = opt.hiddenSize, 
      hierarchy = datasource:hierarchy(),
      root_id = 880542,
      dropout = opt.dropout and nn.Dropout() or nil,
      -- best we can do for now (yet, end of sentences will be under-represented in output updates)
      mvstate = {learn_scale = opt.lrScales[2]/opt.updateInterval},
      --acc_update = opt.accUpdate
      sparse_init = opt.sparseInit
   }
else
   print("Warning: you are using full LogSoftMax for last layer, which "..
      "is really slow (800,000 x hiddenSize multiply adds "..
      "per example. Try --softmaxtree instead.")
   softmax = dp.Neural{
      input_size = opt.hiddenSize,
      output_size = table.length(datasource:classes()),
      transfer = nn.LogSoftMax(),
      dropout = opt.dropout and nn.Dropout() or nil,
      mvstate = {learn_scale = opt.lrScales[2]/opt.updateInterval},
      --acc_update = opt.accUpdate
      sparse_init = opt.sparseInit
   }
end

mlp = dp.Sequential{
   models = {
      dp.RecurrentDictionary{
         dict_size = datasource:vocabularySize(),
         output_size = opt.hiddenSize, rho = opt.rho,
         mvstate = {learn_scale = opt.lrScales[2]}
      },
      softmax
   }
}

--[[Propagators]]--
train = dp.Optimizer{
   loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
   visitor = dp.RecurrentVisitorChain{ -- RNN visitors should be wrapped by this VisitorChain
      visit_interval = opt.updateInterval,
      force_forget = opt.forceForget,
      visitors = {
         dp.Learn{ -- will call nn.Recurrent:updateParameters, which calls nn.Recurrent:backwardThroughTime()
            learning_rate = opt.learningRate, 
            observer = dp.AdaptiveLearningRate{decay_factor=opt.decayFactor, max_wait=opt.maxWait}
         },
         dp.MaxNorm{max_out_norm=opt.maxOutNorm, period=opt.maxNormPeriod}
      }
   },
   feedback = dp.Perplexity(),  
   sampler = dp.SentenceSampler{ 
      evaluate = false,
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}

if not opt.trainOnly then
   valid = dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.SentenceSampler{
         evaluate = true,
         epoch_size = opt.validEpochSize, 
         batch_size = opt.batchSize
      },
      progress = opt.progress
   }
   tester = dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.SentenceSampler{
         evaluate = true,
         batch_size = opt.batchSize
      }
   }
end

--[[Experiment]]--
xp = dp.Experiment{
   model = mlp,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = (not opt.trainOnly) and {
      dp.FileLogger(),
      dp.EarlyStopper{max_epochs = opt.maxTries}
   } or nil,
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

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

print"dp.Models :"
print(mlp)

xp:run(datasource)
