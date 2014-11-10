require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on BillionWords dataset using a Simple Recurrent Neural Network')
cmd:text('Example:')
cmd:text('$> th recurrentlanguagemodel.lua --small --batchSize 512 ')
cmd:text('$> th recurrentlanguagemodel.lua --tiny --batchSize 512 ')
cmd:text('$> th recurrentlanguagemodel.lua --tiny --batchSize 512 --accUpdate --validEpochSize 10000 --trainEpochSize 100000 --softmaxtree')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--decayPoint', 100, 'epoch at which learning rate is decayed')
cmd:option('--decayFactor', 0.1, 'factory by which learning rate is decayed at decay point')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 1, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--batchSize', 512, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons (not recommended)')

cmd:option('--rho', 10, 'back-propagate through time (BPTT) every rho steps')
cmd:option('--hiddenSize', 200, 'number of hidden units used in Simple RNN')

--[[ output layer ]]--
cmd:option('--softmaxtree', false, 'use SoftmaxTree instead of the inefficient (full) softmax')

--[[ data ]]--
cmd:option('--small', false, 'use a small (1/30th) subset of the training set')
cmd:option('--tiny', false, 'use a tiny (1/100th) subset of the training set')
cmd:option('--trainEpochSize', 1000000, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', 100000, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--progress', false, 'print progress bar')

cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)


--[[data]]--
local train_file = 'train_data.th7' 
if opt.small then 
   train_file = 'train_small.th7'
elseif opt.tiny then 
   train_file = 'train_tiny.th7'
end

local datasource = dp.BillionWords{
   context_size = opt.contextSize, train_file = train_file
}

--[[Model]]--
print("Input to first hidden layer has "..
   opt.contextSize*opt.inputEmbeddingSize.." neurons.")

print("Input to second hidden layer has size "..inputSize)

-- build the last layer first:
local softmax
if opt.softmaxtree then
   softmax = dp.SoftmaxTree{
      input_size = opt.hiddenSize, 
      hierarchy = datasource:hierarchy(),
      root_id = 880542,
      dropout = opt.dropout and nn.Dropout() or nil
   }
else
   print("Warning: you are using full LogSoftMax for last layer, which "..
      "is really slow (800,000 x hiddenSize multiply adds "..
      "per example. Try --softmaxtree instead.")
   softmax = dp.Neural{
      input_size = opt.hiddenSize,
      output_size = table.length(datasource:classes()),
      transfer = nn.LogSoftMax(),
      dropout = opt.dropout and nn.Dropout() or nil
   }
end

mlp = dp.Sequential{
   models = {
      dp.RecurrentDictionary{
         dict_size = datasource:vocabularySize(),
         output_size = opt.hiddenSize
      },
      softmax
   }
}

--[[Propagators]]--
train = dp.Optimizer{
   update_interval = opt.rho, -- required for BPTT
   loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
   visitor = {
      dp.Learn{ -- will call nn.Recurrent:updateParameters, which calls nn.Recurrent:backwardThroughTime()
         learning_rate = opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule = {[opt.decayPoint]=opt.learningRate*opt.decayFactor}
         }
      },
      dp.MaxNorm{max_out_norm=opt.maxOutNorm, period=opt.maxNormPeriod}
   },
   feedback = dp.Perplexity(),  
   sampler = dp.SentenceSampler{ 
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}

if not opt.trainOnly then
   valid = dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.SentenceSampler{
         epoch_size = opt.validEpochSize, 
         batch_size = opt.softmaxtree and 1024 or opt.batchSize
      },
      progress = opt.progress
   }
   tester = dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.SentenceSampler{batch_size = opt.softmaxtree and 1024 or opt.batchSize}
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
