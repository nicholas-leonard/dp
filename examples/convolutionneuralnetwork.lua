require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using Convolution Neural Network Training/Optimization')
cmd:text('Example:')
cmd:text('$> th convolutionneuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 2, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--channelSize', '{64,128}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{5,5,5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{2,2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--padding', '{0,0,0,0}', 'amount of zero padding added to the input layer ( should be lower than math.floor(kernelSize/2)') 
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100 | Svhn')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization (recommended)')
cmd:option('--normalInit', false, 'initialize inputs using a normal distribution (as opposed to sparse initialization)')
cmd:option('--activation', 'Tanh', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--hiddenSize', '{}', 'size of the dense hidden layers after the convolution')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.5,0.5}', 'dropout probabilities')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

if opt.activation == 'ReLU' and not opt.normalInit then
   print("Warning : you should probably use --normalInit with ReLUs for "..
      "this script if you don't want to get NaN errors")
end

opt.channelSize = table.fromString(opt.channelSize)
opt.padding = table.fromString(opt.padding)
opt.kernelSize = table.fromString(opt.kernelSize)
opt.kernelStride = table.fromString(opt.kernelStride)
opt.poolSize = table.fromString(opt.poolSize)
opt.poolStride = table.fromString(opt.poolStride)
opt.dropoutProb = table.fromString(opt.dropoutProb)
opt.hiddenSize = table.fromString(opt.hiddenSize)


--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--[[data]]--
local datasource
if opt.dataset == 'Mnist' then
   datasource = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   datasource = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   datasource = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   datasource = dp.Cifar100{input_preprocess = input_preprocess}
elseif opt.dataset == 'Svhn' then
   datasource = dp.Svhn{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end

function dropout(depth)
   return opt.dropout and (opt.dropoutProb[depth] or 0) > 0 and nn.Dropout(opt.dropoutProb[depth])
end

--[[Model]]--

cnn = dp.Sequential()
inputSize = datasource:imageSize('c')
height, width = datasource:imageSize('h'), datasource:imageSize('w')
depth = 1
for i=1,#opt.channelSize do
   local conv = dp.Convolution2D{
      input_size = inputSize, 
      padding = opt.padding[i],
      kernel_size = {opt.kernelSize[i], opt.kernelSize[i]},
      kernel_stride = {opt.kernelStride[i], opt.kernelStride[i]},
      pool_size = {opt.poolSize[i], opt.poolSize[i]},
      pool_stride = {opt.poolStride[i], opt.poolStride[i]},
      output_size = opt.channelSize[i], 
      transfer = nn[opt.activation](),
      dropout = dropout(depth),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit
   }
   cnn:add(conv)
   inputSize, height, width = conv:outputSize(height, width, 'bchw')
   depth = depth + 1
end
inputSize = inputSize*height*width
dp.vprint(not opt.silent, "input to first Neural layer has: "..inputSize.." neurons")

for i,hiddenSize in ipairs(opt.hiddenSize) do
   local dense = dp.Neural{
      input_size = inputSize, 
      output_size = hiddenSize,
      transfer = nn[opt.activation](),
      dropout = dropout(depth),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit
   }
   cnn:add(dense)
   inputSize = hiddenSize
   depth = depth + 1
end

cnn:add(
   dp.Neural{
      input_size = inputSize, 
      output_size = #(datasource:classes()),
      transfer = nn.LogSoftMax(),
      dropout = dropout(depth),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit
   }
)

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
   max_out_norm = opt.maxOutNorm, period=opt.maxNormPeriod
})

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.NLL(),
   visitor = visitor,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = opt.progress
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),
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
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

if not opt.silent then
   print"dp.Models :"
   print(cnn)
   print"nn.Modules :"
   print(cnn:toModule(datasource:trainSet():sub(1,32)))
end
xp:verbose(not opt.silent)

xp:run(datasource)
