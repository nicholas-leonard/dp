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
