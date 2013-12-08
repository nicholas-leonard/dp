------------------------------------------------------------------------
--[[ HyperOptimizer ]]--
--
------------------------------------------------------------------------
local HyperOptimizer = torch.class("dp.HyperOptimizer")
HyperOptimizer.isHyperOptimizer = true

function HyperOptimizer:__init(...)
   local args, id_gen, collection_name, hyperparam_sampler, 
         experiment_factory, datasource_factory 
      = xlua.unpack(
      {... or {}},
      'HyperOptimizer', nil,
      {arg='id_gen', type='dp.EIDGenerator', req=true},
      {arg='collection_name', type='string', req=true,
       help='identifies the collection of experiments'},
      {arg='hyperparam_sampler', type='dp.HyperparamSampler', req=true},
      {arg='experiment_factory', type='dp.ExperimentFactory', req=true},
      {arg='datasource_factory', type='dp.DatasourceFactory', req=true}
   )
   -- experiment id generator
   self._id_gen = id_gen
   self._collection_name = collection_name
   self._hp_sampler = hyperparam_sampler
   self._xp_factory = experiment_factory
   self._ds_factory = datasource_factory
end

function HyperOptimizer:run()
   while true do
      -- sample hyper-parameters 
      local hp = self._hp_sampler:sample()
      print(hp)
      -- assign a unique id
      local id = self._id_gen:nextID()
      -- build datasource
      local ds = self._ds_factory:build(hp)
      -- build experiment
      local xp = self._xp_factory:build(hp, id)
      -- run the experiment on the datasource
      xp:run(ds)
      -- TODO : feedback hyperexperiment to HyperparamSampler
   end
end

------------------------------------------------------------------------
--[[ HyperparamSampler ]]--
-- interface, factory
-- Samples hyper-parameters to initialize and run an experiment
------------------------------------------------------------------------
local HyperparamSampler = torch.class("dp.HyperparamSampler")
HyperparamSampler.isHyperparamSampler = true

function HyperparamSampler:__init(...)
   local args, name = xlua.unpack(
      {... or {}},
      'HyperparamSampler', nil,
      {arg='name', type='string', req=true}
   )
   self._name = name
end

function HyperparamSampler:sample()
   error"NotImplementedError: HyperparamSampler:sample()"
end

------------------------------------------------------------------------
--[[ PriorSampler ]]--
-- factory
-- Samples hyper-parameters to initialize and run an experiment from 
-- a user-provided prior
------------------------------------------------------------------------
local PriorSampler, parent = torch.class("dp.PriorSampler", "dp.HyperparamSampler")
PriorSampler.isPriorSampler = true
      
function PriorSampler:__init(config)
   config = config or {}
   local args, name, dist = xlua.unpack(
      {config},
      'PriorSampler', nil,
      {arg='name', type='string', default='mlp'},
      {arg='dist', type='table'}
   )
   config.name = config.name or name
   parent.__init(self, config)
   self._dist = dist
end

function PriorSampler:sample()
   local hyperparams = {}
   for k, v in pairs(self._dist) do
      if (type(v) == 'table') and v.isChoose then
         hyperparams[k] = v:sample()
      else
         hyperparams[k] = v
      end
   end
   return hyperparams
end

------------------------------------------------------------------------
--[[ ExperimentFactory ]]--
-- interface, factory
-- An experiment factory that can be used to build experiments given
-- a table of hyper-parameters
------------------------------------------------------------------------
local ExperimentFactory = torch.class("dp.ExperimentFactory")
ExperimentFactory.isExperimentFactory = true

function ExperimentFactory:__init(...)
   local args, name = xlua.unpack(
      {... or {}},
      'ExperimentFactory', nil,
      {arg='name', type='string', req=true}
   )
   self._name = name
   -- a cache of objects
   self._cache = {} 
end

function ExperimentFactory:build(hyperparameters, experiment_id)
   error"NotImplementedError : ExperimentFactory:build()"
end

------------------------------------------------------------------------
--[[ DatasourceFactory ]]--
-- interface, factory
-- A datasource factory that can be used to build datasources given
-- a table of hyper-parameters
------------------------------------------------------------------------
local DatasourceFactory = torch.class("dp.DatasourceFactory")
DatasourceFactory.isDatasourceFactory = true

function DatasourceFactory:__init(...)
   local args, name = xlua.unpack(
      {... or {}},
      'DatasourceFactory', nil,
      {arg='name', type='string', req=true}
   )
   self._name = name
   -- a cache of objects
   self._cache = {} 
end

function DatasourceFactory:build(hyperparameters)
   error"NotImplementedError : DatasourceFactory:build()"
end

------------------------------------------------------------------------
--[[ MLPFactory ]]--
-- An example experiment builder for training Mnist using an 
-- MLP of arbitrary dept
------------------------------------------------------------------------
local MLPFactory, parent = torch.class("dp.MLPFactory", "dp.ExperimentFactory")
MLPFactory.isMLPFactory = true
   
function MLPFactory:__init() 
   parent.__init(self, {name='MLP'})
end

function MLPFactory:build(opt, id)
   local function addHidden(mlp, activation, input_size, layer_index)
      layer_index = layer_index or 1
      local output_size = opt.model_width * opt.width_scales[layer_index]
      mlp:add(dp.Linear{input_size=input_size, output_size=output_size})
      if activation == 'ReLU' then
         require 'nnx'
         mlp:add(dp.Module(nn.ReLU()))
      elseif activation == 'Tanh' then
         mlp:add(dp.Module(nn.Tanh()))
      elseif activation == 'Sigmoid' then
         mlp:add(dp.Module(nn.Sigmoid()))
      elseif activation ~= 'Linear' then
         error("Unknown activation function : " .. activation)
      end
      if layer_index < (opt.model_dept-1) then
         addHidden(mlp, activation, output_size, layer_index+1)
      else
         return output_size
      end
   end
   
   --[[Model]]--
   mlp = dp.Sequential()
   -- hidden layer(s)
   local last_size = addHidden(mlp, opt.activation, opt.feature_size, 1)
   -- output layer
   mlp:add(dp.Linear{input_size=last_size, output_size=#opt.classes})
   mlp:add(dp.Module(nn.LogSoftMax()))

   --[[GPU or CPU]]--
   if opt.model_type == 'cuda' then
      require 'cutorch'
      require 'cunn'
      mlp:cuda()
   elseif opt.model_type == 'double' then
      mlp:double()
   elseif opt.model_type == 'float' then
      mlp:float()
   end
   
   --[[ Schedules ]]--
   local schedule
   local lr = opt.learning_rate
   if opt.learning_schedule == '100=/10,200=/10' then 
      schedule={[100]=0.1*lr, [200]=0.01*lr}
   elseif opt.learning_schedule == '100=/10,150=/10' then
      schedule={[100]=0.1*lr, [150]=0.01*lr}
   elseif opt.learning_schedule == '200=/10,250=/10' then
      schedule={[200]=0.1*lr, [250]=0.01*lr}
   elseif opt.learning_schedule == 'none' then
      -- pass
   elseif opt.learning_schedule then
      error("Unknown learning schedule : " .. opt.learning_schedule)
   end
   local lr_schedule 
   if opt.learning_schedule and opt.learning_schedule ~= 'none' then 
      lr_schedule = dp.LearningRateSchedule{schedule=schedule}
   end
   
   --[[ Visitor ]]--
   local visitor = {}
   if opt.momentum and opt.momentum ~= 0 then
      table.insert(visitor, 
         dp.Momentum{
            momentum_factor=opt.momentum, 
            nesterov=opt.nesterov
         }
      )
   end
   if opt.weightdecay and opt.weightdecay ~=0 then
      table.insert(visitor, dp.WeightDecay{wd_factor=opt.weightdecay})
   end
   table.insert(visitor, 
      dp.Learn{
         learning_rate = opt.learning_rate, 
         observer = lr_schedule
      }
   )
   if opt.max_out_norm and opt.max_out_norm ~= 0 then
      table.insert(visitor, dp.MaxNorm{max_out_norm=opt.max_out_norm})
   end

   --[[Propagators]]--
   train = dp.Optimizer{
      criterion = nn.ClassNLLCriterion(),
      observer =  {
         dp.Logger(),
         dp.EarlyStopper{
            error_report = {'validator','feedback','confusion','accuracy'},
            maximize = true,
            max_epochs = opt.max_tries
         }
      },
      visitor = visitor,
      feedback = dp.Confusion(),
      sampler = dp.ShuffleSampler{
         batch_size=opt.batch_size, 
         sample_type=opt.model_type
      },
      progress = true
   }
   valid = dp.Evaluator{
      criterion = nn.ClassNLLCriterion(),
      feedback = dp.Confusion(),  
      observer = dp.Logger(),
      sampler = dp.Sampler{sample_type=opt.model_type}
   }
   test = dp.Evaluator{
      criterion = nn.ClassNLLCriterion(),
      feedback = dp.Confusion(),
      sampler = dp.Sampler{sample_type=opt.model_type}
   }

   --[[Experiment]]--
   return dp.Experiment{
      id = id,
      model = mlp,
      optimizer = train,
      validator = valid,
      tester = test,
      max_epoch = opt.max_epoch
   }
end


------------------------------------------------------------------------
--[[ MnistFactory ]]--
-- interface, factory
-- Builds Mnist datasource instances with common preprocessings
------------------------------------------------------------------------
local MnistFactory, parent = torch.class("dp.MnistFactory", "dp.DatasourceFactory")
MnistFactory.isMnistFactory = true

function MnistFactory:__init(...)
   parent.__init(self, {name='Mnist'})
end

function MnistFactory:build(opt)
   local datasource
   if self._cache[opt.datasource] then
      datasource = self._cache[opt.datasource]
      assert(torch.typename(datasource) and datasource.isMnist)
   elseif opt.datasource == 'mnist' then
      datasource = dp.Mnist()
      self._cache[opt.datasource] = datasource
   elseif opt.datasource == 'mnist:standardize' then
      datasource = dp.Mnist{input_preprocess = dp.Standardize}
      self._cache[opt.datasource] = datasource
   else
      error("Unknown datasource : " .. opt.datasource)
   end
   -- to be used by experiment builder
   opt.feature_size = datasource._feature_size
   opt.classes = datasource._classes
   return datasource
end
