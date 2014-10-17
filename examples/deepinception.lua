require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using Inception Models Training/Optimization')
cmd:text('Example:')
cmd:text('$> th deepinception.lua --lecunlcn --batchSize 128 --accUpdate --cuda')
cmd:text('Options:')
-- fundamentals 
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--activation', 'ReLU', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--batchSize', 32, 'number of examples per batch')
-- regularization (and dropout)
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 1, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}', 'dropout probabilities')
-- data and preprocessing
cmd:option('--dataset', 'Svhn', 'which dataset to use : Svhn | Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization (recommended)')
-- convolution layers
cmd:option('--convChannelSize', '{64,128}', 'Number of output channels (number of filters) for each convolution layer.')
cmd:option('--convKernelSize', '{5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--convKernelStride', '{1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--convPoolSize', '{2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--convPoolStride', '{2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
-- inception layers
cmd:option('--incepChannelSize', '{{32,48},{32,64}}', 'A list of tables of the number of filters in the non-1x1-convolution kernel sizes. Creates an Inception model for each sub-table.')
cmd:option('--incepReduceSize', '{{48,64,32,32},{48,64,32,32}}','Number of filters in the 1x1 convolutions (reduction) '..
   'used in each column. The last 2 are used respectively for the max pooling (projection) column '..
   '(the last column in the paper) and the column that has nothing but a 1x1 conv (the first column in the paper).'..
   'Each subtable should have two elements more than the corresponding outputSize')
cmd:option('--incepReduceStride', '{}', 'The strides of the 1x1 (reduction) convolutions. Defaults to {{1,1,1,..},...}')
cmd:option('--incepKernelSize', '{}', 'The size (height=width) of the non-1x1 convolution kernels. Defaults to {{5,3},...}, i.e. 5x5 and 3x3 for each inception layer')
cmd:option('--incepKernelStride', '{}', 'The stride (height=width) of the convolution. Defaults to {{1,1},...}.')
cmd:option('--incepPoolSize', '{}', 'The size (height=width) of the spatial max pooling used in the next-to-last column. Variables default to 3')
cmd:option('--incepPoolStride', '{}', 'The stride (height=width) of the spatial max pooling. Variables default to 1')
-- dense layers
cmd:option('--hiddenSize', '{}', 'size of the dense hidden layers after the convolution')
-- misc
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate gradients inplace (much faster, but cant be used with momentum')
cmd:option('--progress', false, 'print progress bar')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

-- convolution layers
opt.convChannelSize = table.fromString(opt.convChannelSize)
opt.convKernelSize = table.fromString(opt.convKernelSize)
opt.convKernelStride = table.fromString(opt.convKernelStride)
opt.convPoolSize = table.fromString(opt.convPoolSize)
opt.convPoolStride = table.fromString(opt.convPoolStride)
-- inception layers
opt.incepChannelSize = dp.returnString(opt.incepChannelSize)
opt.incepReduceSize = dp.returnString(opt.incepReduceSize)
opt.incepReduceStride = dp.returnString(opt.incepReduceStride)
opt.incepKernelSize = dp.returnString(opt.incepKernelSize)
opt.incepKernelStride = dp.returnString(opt.incepKernelStride) 
opt.incepPoolSize = table.fromString(opt.incepPoolSize)
opt.incepPoolStride = table.fromString(opt.incepPoolStride)
-- misc
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
if opt.dataset == 'Svhn' then
   datasource = dp.Svhn{input_preprocess = input_preprocess}
elseif opt.dataset == 'Mnist' then
   datasource = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   datasource = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   datasource = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   datasource = dp.Cifar100{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end

--[[Model]]--

cnn = dp.Sequential()
inputSize = datasource:imageSize('c')
height, width = datasource:imageSize('h'), datasource:imageSize('w')
depth = 0
for i=1,#opt.convChannelSize do
   local conv = dp.Convolution2D{
      input_size = inputSize, 
      kernel_size = {opt.convKernelSize[i], opt.convKernelSize[i]},
      kernel_stride = {opt.convKernelStride[i], opt.convKernelStride[i]},
      pool_size = {opt.convPoolSize[i], opt.convPoolSize[i]},
      pool_stride = {opt.convPoolStride[i], opt.convPoolStride[i]},
      output_size = opt.convChannelSize[i], 
      transfer = nn[opt.activation](),
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[i]),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit
   }
   cnn:add(conv)
   inputSize, height, width = conv:outputSize(height, width, 'bchw')
   depth = depth + 1
end

-- Inception layers
for i=1,#opt.incepChannelSize do   
   local incep = dp.Inception{
      input_size = inputSize,
      output_size = opt.incepChannelSize[i],
      reduce_size = opt.incepReduceSize[i],
      reduce_stride = opt.incepReduceStride[i],
      kernel_size = opt.incepKernelSize[i],
      kernel_stride = opt.incepKernelStride[i],
      pool_size = opt.incepPoolSize[i],
      pool_stride =  opt.incepPoolStride[i],
      transfer = nn[opt.activation](),
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[depth]),
      acc_update = opt.accUpdate,
      sparse_init = not opt.normalInit,
      typename = 'inception'..i
   }
   cnn:add(incep)
   inputSize, height, width = incep:outputSize(height, width, 'bchw')
   depth = depth + 1
end

inputSize = inputSize*height*width
print("input to first Neural layer has: "..inputSize.." neurons")

for i,hiddenSize in ipairs(opt.hiddenSize) do
   local dense = dp.Neural{
      input_size = inputSize, 
      output_size = hiddenSize,
      transfer = nn[opt.activation](),
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[depth]),
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
      dropout = opt.dropout and nn.Dropout(opt.dropoutProb[depth]),
      acc_update = opt.accUpdate
   }
)

print(cnn)
print(cnn:toModule(datasource:trainSet():sub(1,32)))

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

xp:run(datasource)

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end
