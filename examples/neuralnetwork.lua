require 'torch'
require 'sys'
require 'dp'
require 'nn'
require 'paths'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--numHidden', 200, 'number of hidden units')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--type', 'double', 'type: double | float | cuda')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons, requires "nnx" luarock')
cmd:option('--dataset', 'Mnist', 'which dataset to use')
cmd:option('--lecunLCN', false, 'apply LeCunLCN preprocessing')
cmd:text()
opt = cmd:parse(arg or {})

print(opt)

--[[Experiment ID generator]]--
id_gen = dp.EIDGenerator('mypc.pid')

--[[preprocessing]]--
local input_preprocess
if opt.lecunLCN then
   input_preprocess=dp.LeCunLCN()
end

--[[Load DataSource]]--
local datasource =dp[opt.dataset]{input_preprocess=input_preprocess}


--[[Model]]--
local dropout
if opt.dropout then
   require 'nnx'
   dropout = nn.Dropout()
end
mlp = dp.Sequential()
mlp:add(dp.Neural{input_size=datasource._feature_size, 
                  output_size=opt.numHidden,
                  transfer=nn.Tanh()})
mlp:add(dp.Neural{input_size=opt.numHidden, 
                  output_size=#(datasource._classes),
                  transfer=nn.LogSoftMax(),
                  dropout=dropout})

--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   mlp:cuda()
end

--[[Propagators]]--
train = dp.Optimizer{
   criterion = nn.ClassNLLCriterion(),
   visitor = { -- the ordering here is important:
      dp.Momentum{momentum_factor=opt.momentum},
      dp.Learn{
         learning_rate=opt.learningRate, 
         observer = dp.LearningRateSchedule{
            schedule={[200]=0.01, [400]=0.001}
         }
      },
      dp.MaxNorm{max_out_norm=opt.maxOutNorm}
   },
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size=opt.batchSize, sample_type=opt.type},
   progress = true
}
valid = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{sample_type=opt.type}
}
test = dp.Evaluator{
   criterion = nn.ClassNLLCriterion(),
   feedback = dp.Confusion(),
   sampler = dp.Sampler{sample_type=opt.type}
}

--[[Experiment]]--
xp = dp.Experiment{
   id_gen = id_gen,
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
