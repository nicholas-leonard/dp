require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Facial Keypoint detector using Convolution Neural Network Training/Optimization')
cmd:text('Example:')
cmd:text('$> th facialkeypointdetector.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--padding', false, 'add math.floor(kernelSize/2) padding to the input of each convolution') 
cmd:option('--channelSize', '{64,96,96}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{5,5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--hiddenSize', '{3000}', 'size of the dense hidden layers (after convolutions, before output)')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--activation', 'ReLU', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.5,0.5}', 'dropout probabilities')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--submissionFile', '', 'Kaggle submission will be saved to a file with this name')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--validRatio', 1/10, 'proportion of dataset used for cross-validation')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

assert(opt.submissionFile ~= '', 'provide filename, e.g.: --submissionFile submission12.csv')

opt.channelSize = table.fromString(opt.channelSize)
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

--[[data]]--
ds = dp.FacialKeypoints{input_preprocess = input_preprocess, valid_ratio = opt.validRatio}

--[[Model]]--
cnn = nn.Sequential()

-- convolutional and pooling layers
inputSize = ds:imageSize('c')
depth = 1
for i=1,#opt.channelSize do
   if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn:add(nn.SpatialDropout(opt.dropoutProb[depth]))
   end
   cnn:add(nn.SpatialConvolution(
      inputSize, opt.channelSize[i], 
      opt.kernelSize[i], opt.kernelSize[i], 
      opt.kernelStride[i], opt.kernelStride[i],
      opt.padding and math.floor(opt.kernelSize[i]/2) or 0
   ))
   if opt.batchNorm then
      -- batch normalization can be awesome
      cnn:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
   end
   cnn:add(nn[opt.activation]())
   if opt.poolSize[i] and opt.poolSize[i] > 0 then
      cnn:add(nn.SpatialMaxPooling(
         opt.poolSize[i], opt.poolSize[i], 
         opt.poolStride[i] or opt.poolSize[i], 
         opt.poolStride[i] or opt.poolSize[i]
      ))
   end
   inputSize = opt.channelSize[i]
   depth = depth + 1
end
-- get output size of convolutional layers
outsize = cnn:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
inputSize = outsize[2]*outsize[3]*outsize[4]
dp.vprint(not opt.silent, "input to dense layers has: "..inputSize.." neurons")

cnn:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)

-- dense hidden layers
cnn:add(nn.Collapse(3))
for i,hiddenSize in ipairs(opt.hiddenSize) do
   if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
      cnn:add(nn.Dropout(opt.dropoutProb[depth]))
   end
   cnn:add(nn.Linear(inputSize, hiddenSize))
   if opt.batchNorm then
      cnn:add(nn.BatchNormalization(hiddenSize))
   end
   cnn:add(nn[opt.activation]())
   inputSize = hiddenSize
   depth = depth + 1
end

-- output layer
if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
   cnn:add(nn.Dropout(opt.dropoutProb[depth]))
end
cnn:add(nn.Linear(inputSize, 30*98))
-- we use nn.MultiSoftMax() Module for detecting coordinates :
cnn:add(nn.Reshape(30,98))
cnn:add(nn.MultiSoftMax())

--[[Propagators]]--
baseline = ds:loadBaseline()

logModule = nn.Sequential()
logModule:add(nn.AddConstant(0.00000001)) -- fixes log(0)=NaN errors
logModule:add(nn.Log())

train = dp.Optimizer{
   acc_update = opt.accUpdate,
   loss = nn.ModuleCriterion(nn.DistKLDivCriterion(), logModule, nn.Convert()),
   callback = function(model, report) 
      -- the ordering here is important
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   feedback = dp.FacialKeypointFeedback{baseline=baseline, precision=98},
   progress = opt.progress
}
valid = dp.Evaluator{
   sampler = dp.Sampler{batch_size = opt.batchSize},
   feedback = dp.FacialKeypointFeedback{baseline=baseline, precision=98}
}
test = dp.Evaluator{
   feedback = dp.FKDKaggle{
      submission = ds:loadSubmission(), 
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

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Models :"
   print(cnn)
end

print(ds)
xp:run(ds)
