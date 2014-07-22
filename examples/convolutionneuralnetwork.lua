require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using Convolutional Neural Network Training/Optimization')
cmd:text('Example:')
cmd:text('$> th convolutionneuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--channelSize', '{64,128}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

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
if opt.dataset == 'Mnist' then
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
local dropout
if opt.dropout then
   dropout = nn.Dropout()
end

mlp = dp.Sequential{
   models = {
      dp.Convolution2D{
         input_size = datasource:imageSize('c'), 
         kernel_size = {opt.kernelSize[1], opt.kernelSize[1]},
         kernel_stride = {opt.kernelStride[1], opt.kernelStride[1]},
         pool_size = {opt.poolSize[1], opt.poolSize[1]},
         pool_stride = {opt.poolStride[1], opt.poolStride[1]},
         output_size = opt.channelSize[1], 
         transfer = nn.Tanh()
      },
      dp.Convolution2D{
         input_size = opt.channelSize[1], 
         kernel_size = {opt.kernelSize[2], opt.kernelSize[2]},
         kernel_stride = {opt.kernelStride[2], opt.kernelStride[2]},
         pool_size = {opt.poolSize[2], opt.poolSize[2]},
         pool_stride = {opt.poolStride[2], opt.poolStride[2]},
         output_size = opt.channelSize[2], 
         transfer = nn.Tanh()
      },
      dp.Neural{
         input_size = opt.nHidden, 
         output_size = #(datasource:classes()),
         transfer = nn.LogSoftMax(),
         dropout = dropout
      }
   }
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   mlp:cuda()
end

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.NLL(),
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor = opt.momentum},
      dp.Learn{
         learning_rate = opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule = {[200]=0.01, [400]=0.001}
         }
      },
      dp.MaxNorm{max_out_norm = opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler()
}
test = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),
   sampler = dp.Sampler()
}

--[[Experiment]]--
xp = dp.Experiment{
   model = mlp,
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
