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

function DatasourceFactory:name()
   return self._name
end

function DatasourceFactory:hyperReport()
   return {name = self._name}
end 
