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
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--activation', 'Tanh', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.5,0.5}', 'dropout probabilities')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

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
   datasource = dp.FacialKeypoints{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end

--[[Model]]--

cnn = dp.Sequential()
inputSize = datasource:imageSize('c')
outputSize = {datasource:imageSize('h'), datasource:imageSize('w')}
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
      acc_update = opt.accUpdate
   }
   cnn:add(conv)
   inputSize = opt.channelSize[i]
   outputSize[1] = conv:nOutputFrame(outputSize[1], 1)
   outputSize[2] = conv:nOutputFrame(outputSize[2], 2)
end

inputSize = inputSize
cnn:add(
   dp.Neural{
      input_size = inputSize*outputSize[1]*outputSize[2], 
      output_size = opt.hiddenSize,
      transfer = nn[opt.activation](),
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[#opt.channelSize]),
      acc_update = opt.accUpdate
   }
)

-- we use a special nn.MultiSoftMax() Module for detecting coordinates :
local multisoftmax = nn.Sequential()
multisoftmax:add(nn.Reshape(30,98))
multisoftmax:add(nn.MultiSoftMax())
multisoftmax:add(nn.Log())
cnn:add(
   dp.Neural{
      input_size = opt.hiddenSize, 
      output_size = 30*98,
      transfer = multisoftmax,
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[#opt.channelSize]),
      acc_update = opt.accUpdate
   }
)

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
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

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.DistKLDivCriterion(),
   visitor = visitor,
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   loss = dp.DistKLDivCriterion(),
   sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{
   loss = dp.Null(), -- because we don't have targets for the test set
   feedback = dp.FKDKaggle{submission=datasource:loadSubmission()},
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
      dp.EarlyStopper{max_epochs = opt.maxTries}
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

xp:run(datasource)
