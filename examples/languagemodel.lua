require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on BillionWords dataset using SoftmaxTree')
cmd:text('Example:')
cmd:text('$> th languagemodel.lua --small --batchSize 512 ')
cmd:text('$> th languagemodel.lua --tiny --batchSize 512 ')
cmd:text('$> th languagemodel.lua --tiny --batchSize 512 --accUpdate --validEpochSize 10000 --trainEpochSize 100000 --softmaxtree')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--decayPoint', 100, 'epoch at which learning rate is decayed')
cmd:option('--decayFactor', 0.1, 'factory by which learning rate is decayed at decay point')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 5, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--batchSize', 512, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--accUpdate', false, 'accumulate updates inplace using accUpdateGradParameters')

cmd:option('--contextSize', 5, 'number of words preceding the target word used to predict the target work')
cmd:option('--inputEmbeddingSize', 100, 'number of neurons per word embedding')

--[[ first hidden layer ]]--
cmd:option('--neuralSize', 200, 'number of hidden units used for first hidden layer (used when --convolution is not used)')
--or
cmd:option('--convolution', false, 'use a Convolution1D instead of Neural for the first hidden layer')
cmd:option('--convOutputSize', 200, 'number of output neurons of the convolutional kernel (outputFrameSize)')
cmd:option('--convKernelSize', 2, 'number of words considered by convolution')
cmd:option('--convKernelStride', 1, 'stride (step size) of the convolution')
cmd:option('--convPoolSize', 2, 'number of words max pooled after convolution')
cmd:option('--convPoolStride', 2, 'stride of the max pooling after the convolution') 

cmd:option('--outputEmbeddingSize', 100, 'number of hidden units at softmaxtree')

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

cmd:text()
opt = cmd:parse(arg or {})
print(opt)


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

local hiddenModel, inputSize
if opt.convolution then
   print"Using convolution for first hidden layer"
   hiddenModel = dp.Convolution1D{
      input_size = opt.inputEmbeddingSize, 
      output_size = opt.convOutputSize,
      kernel_size = opt.convKernelSize,
      kernel_stride = opt.convKernelStride,
      pool_size = opt.convPoolSize,
      pool_stride = opt.convPoolStride,
      transfer = nn.Tanh(),
      dropout = opt.dropout and nn.Dropout() or nil,
      acc_update = opt.accUpdate
   }
   local nOutputFrame = hiddenModel:nOutputFrame(opt.contextSize)
   print("Convolution has "..nOutputFrame.." output Frames")
   inputSize = nOutputFrame*opt.convOutputSize
else
   hiddenModel = dp.Neural{
      input_size = opt.contextSize*opt.inputEmbeddingSize,
      output_size = opt.neuralSize, 
      transfer = nn.Tanh(),
      dropout = opt.dropout and nn.Dropout() or nil,
      acc_update = opt.accUpdate
   }
   inputSize = opt.neuralSize
end

print("Input to second hidden layer has size "..inputSize)

local softmax
if opt.softmaxforest then
   softmax = dp.SoftmaxForest{
      input_size = opt.outputEmbeddingSize, 
      hierarchy = {  
         datasource:hierarchy('word_tree1.th7'), 
         datasource:hierarchy('word_tree2.th7'),
         datasource:hierarchy('word_tree3.th7')
      },
      gater_size = table.fromString(opt.forestGaterSize),
      gater_act = nn.Tanh(),
      root_id = {880542,880542,880542},
      dropout = opt.dropout and nn.Dropout() or nil,
      acc_update = opt.accUpdate
   }
   opt.softmaxtree = true
elseif opt.softmaxtree then
   softmax = dp.SoftmaxTree{
      input_size = opt.outputEmbeddingSize, 
      hierarchy = datasource:hierarchy(),
      root_id = 880542,
      dropout = opt.dropout and nn.Dropout() or nil,
      acc_update = opt.accUpdate
   }
else
   print("Warning: you are using full LogSoftMax for last layer, which "..
      "is really slow (800,000 x outputEmbeddingSize multiply adds "..
      "per example. Try --softmaxtree instead.")
   softmax = dp.Neural{
      input_size = opt.outputEmbeddingSize,
      output_size = table.length(datasource:classes()),
      transfer = nn.LogSoftMax(),
      dropout = opt.dropout and nn.Dropout() or nil,
      acc_update = opt.accUpdate
   }
end

mlp = dp.Sequential{
   models = {
      dp.Dictionary{
         dict_size = datasource:vocabularySize(),
         output_size = opt.inputEmbeddingSize,
         acc_update = opt.accUpdate
      },
      hiddenModel,
      dp.Neural{
         input_size = inputSize, 
         output_size = opt.outputEmbeddingSize, 
         transfer = nn.Tanh(),
         dropout = opt.dropout and nn.Dropout() or nil,
         acc_update = opt.accUpdate
      },
      softmax
   }
}

--[[GPU or CPU]]--
if opt.cuda then
   print"Using CUDA"
   require 'cutorch'
   require 'cunn'
   if opt.softmaxtree then
      require 'cunnx'
   end
   cutorch.setDevice(opt.useDevice)
   mlp:cuda()
end

--[[Propagators]]--
train = dp.Optimizer{
   loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
   visitor = {
      dp.Learn{
         learning_rate = opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule = {[opt.decayPoint]=opt.learningRate*opt.decayFactor}
         }
      },
      dp.MaxNorm{max_out_norm=opt.maxOutNorm, period=opt.maxNormPeriod}
   },
   feedback = dp.Perplexity(),  
   sampler = dp.Sampler{ --shuffle sample takes too much mem
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}

if not opt.trainOnly then
   valid = dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{
         epoch_size = opt.validEpochSize, 
         batch_size = opt.softmaxtree and 1024 or opt.batchSize
      },
      progress = opt.progress
   }
   tester = dp.Evaluator{
      loss = opt.softmaxtree and dp.TreeNLL() or dp.NLL(),
      feedback = dp.Perplexity(),  
      sampler = dp.Sampler{batch_size = opt.softmaxtree and 1024 or opt.batchSize}
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

xp:run(datasource)
