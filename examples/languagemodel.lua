require 'dp'
--error"Work in progress: not ready for use"

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on BillionWords dataset using SoftmaxTree')
cmd:text('Example:')
cmd:text('$> th languagemodel.lua --small --batchSize 512 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--batchSize', 512, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--embeddingSize', 100, 'number of neurons per word embedding')
cmd:option('--contextSize', 5, 'number of words preceding the target word used to predict the target work')
cmd:option('--convOutputSize', 200, 'number of output neurons of the convolutional kernel (outputFrameSize)')
cmd:option('--convKernelSize', 2, 'number of words considered by convolution')
cmd:option('--convKernelStride', 1, 'stride (step size) of the convolution')
cmd:option('--convPoolSize', 2, 'number of words max pooled after convolution')
cmd:option('--convPoolStride', 2, 'stride of the max pooling after the convolution') 
cmd:option('--nHidden', 200, 'number of hidden units at softmaxtree')
cmd:option('--small', false, 'use a small (1/30th) subset of the training set')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)


--[[data]]--
local datasource 
if opt.small then 
   datasource = dp.BillionWords{
      context_size = opt.contextSize, train_file = 'train_small.th7'
   }
else
   datasource = dp.BillionWords{context_size = opt.contextSize}
end

--[[Model]]--
local dropout
if opt.dropout then
   require 'nnx'
end

-- measure number of output frames of the convolution1D layer
local nFrame = (opt.contextSize - opt.convKernelSize) / opt.convKernelStride + 1
nFrame = (nFrame - opt.convPoolSize) / opt.convPoolStride + 1

mlp = dp.Sequential{
   models = {
      dp.Dictionary{
         dict_size = datasource:vocabularySize(),
         output_size = opt.embeddingSize
      },
      dp.Convolution1D{
         input_size = opt.embeddingSize, 
         output_size = opt.convOutputSize,
         kernel_size = opt.convKernelSize,
         kernel_stride = opt.convKernelStride,
         pool_size = opt.convPoolSize,
         pool_stride = opt.convPoolStride,
         dropout = opt.dropout and nn.Dropout() or nil,
         transfer = nn.Tanh()
      },
      dp.Neural{
         input_size = nFrame*opt.convOutputSize, 
         output_size = opt.nHidden, 
         transfer = nn.Tanh(),
         dropout = opt.dropout and nn.Dropout() or nil
      },
      dp.SoftmaxTree{
         input_size = opt.nHidden, 
         hierarchy = datasource:hierarchy(),
         dropout = opt.dropout and nn.Dropout() or nil
      }
   }
}

--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   mlp:cuda()
end

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.TreeNLL(),
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
   --feedback = dp.Perplexity(),  
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   loss = dp.TreeNLL(),
   sampler = dp.Sampler()
}
test = dp.Evaluator{
   loss = dp.TreeNLL(),
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
      --[[dp.EarlyStopper{
         maximize = true,
         max_epochs = opt.maxTries
      }--]]
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

xp:run(datasource)
