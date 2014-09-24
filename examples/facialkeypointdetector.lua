require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Facial Keypoint detector using Convolution Neural Network Training/Optimization')
cmd:text('Example:')
cmd:text('$> th facialkeypointdetector.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 2, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--channelSize', '{64,128}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--hiddenSize', 1000, 'size of the dense hidden layer (after convolutions, before output)')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--dataset', 'FacialKeypoints', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--activation', 'ReLU', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.5,0.5}', 'dropout probabilities')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--submissionFile', '', 'Kaggle submission will be saved to a file with this name')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--normalInit', false, 'initialize inputs using a normal distribution (as opposed to sparse initialization)')
cmd:option('--validRatio', 1/10, 'proportion of dataset used for cross-validation')
cmd:option('--neuralSize', 1000, 'Size of first neural layer in 3 Neural Layer MLP.')
cmd:option('--mlp', false, 'use multi-layer perceptron, as opposed to convolution neural network')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

assert(opt.submissionFile ~= '', 'provide filename, e.g.: --submissionFile submission12.csv')

opt.channelSize = table.fromString(opt.channelSize)
opt.kernelSize = table.fromString(opt.kernelSize)
opt.kernelStride = table.fromString(opt.kernelStride)
opt.poolSize = table.fromString(opt.poolSize)
opt.poolStride = table.fromString(opt.poolStride)
opt.dropoutProb = table.fromString(opt.dropoutProb)

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end

--[[data]]--
local datasource
if opt.dataset == 'FacialKeypoints' then
   datasource = dp.FacialKeypoints{
      input_preprocess = input_preprocess, valid_ratio = opt.validRatio
   }
else
    error("Unknown Dataset")
end

--[[Model]]--

cnn = dp.Sequential()

local inputSize
if not opt.mlp then
   inputSize = datasource:imageSize('c')
   height, width = datasource:imageSize('h'), datasource:imageSize('w')
   for i=1,#opt.channelSize do
      local conv = dp.Convolution2D{
         input_size = inputSize, 
         kernel_size = {opt.kernelSize[i], opt.kernelSize[i]},
         kernel_stride = {opt.kernelStride[i], opt.kernelStride[i]},
         pool_size = {opt.poolSize[i], opt.poolSize[i]},
         pool_stride = {opt.poolStride[i], opt.poolStride[i]},
         output_size = opt.channelSize[i], 
         transfer = nn[opt.activation](),
         dropout = opt.dropout and nn.Dropout(opt.dropoutProb[i]),
         acc_update = opt.accUpdate,
         sparse_init = not opt.normalInit
      }
      cnn:add(conv)
      inputSize, height, width = conv:outputSize(height, width, 'bchw')
   end
   inputSize = inputSize*height*width
   print("input to first Neural layer has: "..inputSize.." neurons")
else
   inputSize = datasource:featureSize()
   if opt.neuralSize > 0 then
      cnn:add(
         dp.Neural{
            input_size = inputSize, 
            output_size = opt.neuralSize,
            transfer = nn[opt.activation](),
            dropout = opt.dropout and nn.Dropout(opt.dropoutProb[#opt.channelSize]),
            acc_update = opt.accUpdate,
            sparse_init = not opt.normalInit
         }
      )
      inputSize = opt.neuralSize
   end
end

cnn:add(
   dp.Neural{
      input_size = inputSize, 
      output_size = opt.hiddenSize,
      transfer = nn[opt.activation](),
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[#opt.channelSize]),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit
   }
)

-- we use a special nn.MultiSoftMax() Module for detecting coordinates :
local multisoftmax = nn.Sequential()
multisoftmax:add(nn.Reshape(30,98))
multisoftmax:add(nn.MultiSoftMax())
cnn:add(
   dp.Neural{
      input_size = opt.hiddenSize, 
      output_size = 30*98,
      transfer = multisoftmax,
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[#opt.channelSize]),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit,
      output_view = 'bwc', -- because of the multisoftmax,
      output = dp.SequenceView() --same
   }
)

print(cnn)

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   cnn:cuda()
end

local visitor = {}
-- the ordering here is important:
if opt.momentum > 0 then
   if opt.accUpdate then
      error"momentum doesn't work with --accUpdate"
   end
   table.insert(visitor, dp.Momentum{momentum_factor = opt.momentum})
end
table.insert(visitor, dp.Learn{learning_rate = opt.learningRate})
table.insert(visitor, dp.MaxNorm{
   max_out_norm = opt.maxOutNorm, period = opt.maxNormPeriod
})

local baseline = datasource:loadBaseline()
local logModule = nn.Sequential()
logModule:add(nn.AddConstant(0.00000001)) -- fixes log(0)=NaN errors
logModule:add(nn.Log())

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.KLDivergence{input_module=logModule:clone()},
   visitor = visitor,
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = opt.progress,
   feedback = dp.FacialKeypointFeedback{baseline=baseline, precision=98}
}
valid = dp.Evaluator{
   loss = dp.KLDivergence{input_module=logModule:clone()},
   sampler = dp.Sampler{batch_size = opt.batchSize},
   feedback = dp.FacialKeypointFeedback{baseline=baseline, precision=98}
}
test = dp.Evaluator{
   loss = dp.Null(), -- because we don't have targets for the test set
   feedback = dp.FKDKaggle{
      submission = datasource:loadSubmission(), 
      file_name = opt.submissionFile
   },
   sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--
xp = dp.Experiment{
   model = cnn,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','facialkeypoint','mse'},
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

xp:run(datasource)
