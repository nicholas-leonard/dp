require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using Mixture of Experts Training/Optimization')
cmd:text('Example:')
cmd:text('$> th mixtureofexperts.lua --batchSize 128 --momentum 0.9')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--schedule', '{[200]=0.01, [400]=0.001}', 'learning rate schedule')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--nExpert', 3, 'number of experts')
cmd:option('--expertSize', '{128,128}', 'number of hidden units per expert')
cmd:option('--gaterSize', '{64,64}', 'number of hidden units in gater')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--progress', false, 'print progress bar')
cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.expertSize = table.fromString(opt.expertSize)
opt.gaterSize = table.fromString(opt.gaterSize)
if not opt.silent then
   table.print(opt)
end

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end

--[[data]]--
local ds
if opt.dataset == 'Mnist' then
   ds = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   ds = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   ds = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   ds = dp.Cifar100{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end

--[[Model]]--

-- experts
experts = nn.ConcatTable()

for i=1,opt.nExpert do
   local inputSize = ds:featureSize()
   local expert = nn.Sequential()
   for i,hiddenSize in ipairs(opt.expertSize) do
      expert:add(nn.Linear(inputSize, hiddenSize))
      expert:add(nn.Tanh())
      inputSize = hiddenSize
   end
   expert:add(nn.Linear(inputSize, #(ds:classes())))
   expert:add(nn.LogSoftMax())
   experts:add(expert)
end

-- gater
gater = nn.Sequential()

inputSize = ds:featureSize()
for i,hiddenSize in ipairs(opt.gaterSize) do
   gater:add(nn.Linear(inputSize, hiddenSize))
   gater:add(nn.Tanh())
   inputSize = hiddenSize
end
gater:add(nn.Linear(inputSize, opt.nExpert))
gater:add(nn.SoftMax())

-- mixture of experts
moe = nn.Sequential()
moe:add(nn.Convert(ds:ioShapes(), 'bf'))

trunk = nn.ConcatTable()
trunk:add(gater)
trunk:add(experts)

moe:add(trunk)
moe:add(nn.MixtureTable())

--[[Propagators]]--
train = dp.Optimizer{
   loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
   callback = function(model, report) 
      opt.learningRate = opt.schedule[report.epoch] or opt.learningRate
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = opt.progress
}
valid = dp.Evaluator{
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{
   feedback = dp.Confusion(),
   sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--
xp = dp.Experiment{
   model = moe,
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
xp:verbose(not opt.silent)
if not opt.silent then
   print"Model :"
   print(moe)
end

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:run(ds)
