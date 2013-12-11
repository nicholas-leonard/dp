------------------------------------------------------------------------
--[[ HyperOptimizer ]]--
-- Samples hyperparameters from a HyperparamSampler and 
-- passes these to a DatasourceFactory and ExperimentFactory to 
-- generate an Experiment wich is run until it terminates (returns) and
-- then the cyle starts again.
------------------------------------------------------------------------
local PGHyperOptimizer, parent = torch.class("dp.PGHyperOptimizer", "dp.HyperOptimizer")
PGHyperOptimizer.isPGHyperOptimizer = true

function PGHyperOptimizer:__init(config)
   config = config or {}
   local args, id_gen, collection_name, hyperparam_sampler, 
         experiment_factory, datasource_factory 
      = xlua.unpack(
      {... or {}},
      'HyperOptimizer', nil,
      {arg='id_gen', type='dp.PGEIDGenerator', req=true},
      {arg='pg', type='dp.Postgres', help='defaults to dp.Postgres()'}
   )
   config.id_gen = id_gen
   self._pg = pg or dp.Postgres()
   parent.__init(self, config)
end

function PGHyperOptimizer:hyperReport(id, hyperparam)
   return {
      collection_name = self._collection_name,
      hyperparam_sampler = self._hp_sampler:hyperReport(),
      experiment_factory = self._xp_factory:hyperReport(),
      datasource_factory = self._ds_factory:hyperReport(),
      
   }
end

function PGHyperOptimizer:run()
   while true do
      -- sample hyperparameters 
      local hp = self._hp_sampler:sample()
      -- assign a unique id
      local id = self._id_gen:nextID()
      -- build datasource
      local ds = self._ds_factory:build(hp)
      -- build experiment
      local xp = self._xp_factory:build(hp, id)
      -- log hyperparameters
      self:logHyperparams(id, hp)
      -- run the experiment on the datasource
      xp:run(ds)
      -- TODO : feedback hyperexperiment to HyperparamSampler
   end
end
