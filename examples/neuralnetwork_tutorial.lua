require 'dp'

--[[hyperparameters]]--
opt = {
   nHidden = 100, --number of hidden units
   learningRate = 0.1, --training learning rate
   momentum = 0.9, --momentum factor to use for training
   maxOutNorm = 1, --maximum norm allowed for output neuron weights
   batchSize = 128, --number of examples per mini-batch
   maxTries = 100, --maximum number of epochs without reduction in validation error.
   maxEpoch = 1000 --maximum number of epochs of training
}

--[[data]]--
datasource = dp.Mnist{input_preprocess = dp.Standardize()}

--[[Model]]--
model = nn.Sequential():extend(
   nn.Convert(datasource:ioShapes(), 'bf'), -- to batchSize x nFeature (also type converts)
   nn.Linear(datasource:featureSize(), opt.nHidden), 
   nn.Tanh(),
   nn.Linear(opt.nHidden, #(datasource:classes())),
   nn.LogSoftMax()
)
model:zeroGradParameters() -- don't forget this, else NaN errors

--[[Propagators]]--
train = dp.Optimizer{
   loss = nn.ClassNLLCriterion(),
   callback = function(model, report) 
      -- the ordering here is important
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = true
}
valid = dp.Evaluator{
   feedback = dp.Confusion(),  
   sampler = dp.Sampler()
}
test = dp.Evaluator{
   feedback = dp.Confusion(),
   sampler = dp.Sampler()
}

--[[Experiment]]--
xp = dp.Experiment{
   model = model,
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
