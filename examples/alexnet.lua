require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training ImageNet (large-scale image classification) using an Alex Krizhevsky Convolution Neural Network')
cmd:test('Ref.: A. http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf')
cmd:test('B. https://github.com/facebook/fbcunn/blob/master/examples/imagenet/models/alexnet_cunn.lua')
cmd:text('Example:')
cmd:text('$> th alexnet.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 2, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0, 'momentum') 
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--progress', false, 'print progress bar')
cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)

opt.channelSize = table.fromString(opt.channelSize)
opt.padding = table.fromString(opt.padding)
opt.kernelSize = table.fromString(opt.kernelSize)
opt.kernelStride = table.fromString(opt.kernelStride)
opt.poolSize = table.fromString(opt.poolSize)
opt.poolStride = table.fromString(opt.poolStride)
opt.dropoutProb = table.fromString(opt.dropoutProb)
opt.hiddenSize = table.fromString(opt.hiddenSize)


--[[data]]--
datasource = dp.ImageNet()

--[[Model]]--
-- We create the model using pure nn
function createModel()
   local features = nn.Concat(2)
   local fb1 = nn.Sequential() -- branch 1
   fb1:add(nn.SpatialConvolutionMM(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   
   fb1:add(nn.SpatialConvolutionMM(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   
   fb1:add(nn.SpatialConvolutionMM(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionMM')) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   
   classifier:add(nn.Linear(4096, 1000))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model
end

-- wrap the nn.Module in a dp.Module
model = dp.Module{
   module = createModel(),
   input_view = 'bchw', 
   output_view = 'bf',
   output = dp.ClassView()
}

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
   sampler = dp.RandomSampler{batch_size = opt.batchSize},
   progress = opt.progress
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.TopCrop(),  
   sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--
xp = dp.Experiment{
   model = cnn,
   optimizer = train,
   validator = valid,
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

print"dp.Models :"
print(cnn)
print"nn.Modules :"
print(cnn:toModule(datasource:trainSet():sub(1,32)))

xp:run(datasource)
