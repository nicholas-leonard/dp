------------------------------------------------------------------------
--[[ HyperOptimizer ]]--
-- Samples hyperparameters from a HyperparamSampler and 
-- passes these to a DatasourceFactory and ExperimentFactory to 
-- generate an Experiment wich is run until it terminates (returns) and
-- then the cyle starts again.
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
