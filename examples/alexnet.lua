require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training ImageNet (large-scale image classification) using an Alex Krizhevsky Convolution Neural Network')
cmd:text('Ref.: A. http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf')
cmd:text('B. https://github.com/facebook/fbcunn/blob/master/examples/imagenet/models/alexnet_cunn.lua')
cmd:text('Example:')
cmd:text('$> th alexnet.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--dataPath', paths.concat(dp.DATA_DIR, 'ImageNet'), 'path to ImageNet')
cmd:option('--trainPath', '', 'Path to train set. Defaults to --dataPath/ILSVRC2012_img_train')
cmd:option('--validPath', '', 'Path to valid set. Defaults to --dataPath/ILSVRC2012_img_val')
cmd:option('--metaPath', '', 'Path to metadata. Defaults to --dataPath/metadata')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
 cmd:option('-weightDecay', 5e-4, 'weight decay')
cmd:option('--maxNormPeriod', 1, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0.9, 'momentum') 
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--nThread', 2, 'allocate threads for loading images from disk. Requires threads-ffi.')
cmd:option('--LCN', false, 'use Local Constrast Normalization as in the original paper. Requires inn (imagine-nn)')
cmd:text()
opt = cmd:parse(arg or {})

opt.trainPath = (opt.trainPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_train') or opt.trainPath
opt.validPath = (opt.validPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_val') or opt.validPath
opt.metaPath = (opt.metaPath == '') and paths.concat(opt.dataPath, 'metadata') or opt.metaPath
table.print(opt)

if opt.LCN then
   assert(opt.cuda, "LCN only works with CUDA")
   require "inn"
end


--[[data]]--
datasource = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath, 
   meta_path=opt.metaPath, verbose=opt.verbose
}

-- preprocessing function 
ppf = datasource:normalizePPF()

--[[Model]]--
-- We create the model using pure nn
function createModel()
   local features = nn.Concat(2)
   local fb1 = nn.Sequential() -- branch 1
   fb1:add(nn.SpatialConvolutionMM(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(nn.ReLU())
   if opt.LCN then
      fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
   end
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   
   fb1:add(nn.SpatialConvolutionMM(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(nn.ReLU())
   if opt.LCN then
      fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
   end
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   
   fb1:add(nn.SpatialConvolutionMM(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   fb1:add(nn.Copy(nil, nil, true)) -- prevents a newContiguous in SpatialMaxPooling:backward()

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

--[[Visitor]]--
local visitor = {}
-- the ordering here is important:
if opt.momentum > 0 then
   if opt.accUpdate then
      print"Warning : momentum is ignored with acc_update = true"
   end
   table.insert(visitor, 
      dp.Momentum{momentum_factor = opt.momentum}
   )
end
if opt.weightDecay and opt.weightDecay > 0 then
   if opt.accUpdate then
      print"Warning : weightdecay is ignored with acc_update = true"
   end
   table.insert(visitor, dp.WeightDecay{wd_factor=opt.weightDecay})
end
table.insert(visitor, 
   dp.Learn{
      learning_rate = opt.learningRate, 
      observer = dp.LearningRateSchedule{
         schedule={[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4}
      }
   }
)
if opt.maxOutNorm > 0 then
   table.insert(visitor, dp.MaxNorm{
      max_out_norm = opt.maxOutNorm, period=opt.maxNormPeriod
   })
end

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.NLL(),
   visitor = visitor,
   feedback = dp.Confusion(),
   sampler = dp.RandomSampler{
      batch_size=opt.batchSize, epoch_size=opt.trainEpochSize, ppf=ppf
   },
   progress = opt.progress
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.TopCrop{n_top={1,5,10},n_crop=10,center=2},  
   sampler = dp.Sampler{
      batch_size=math.round(opt.batchSize/10),
      ppf=ppf
   }
}

--[[multithreading]]--
if opt.nThread > 0 then
   datasource:multithread(opt.nThread)
   train:sampler():async()
   valid:sampler():async()
end

--[[Experiment]]--
xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','topcrop','all',5},
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

print"nn.Modules :"
print(model:toModule(datasource:trainSet():sub(1,32)))

xp:run(datasource)
