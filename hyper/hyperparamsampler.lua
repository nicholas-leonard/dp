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

function HyperparamSampler:name()
   return self._name
end


function HyperparamSampler:hyperReport()
   return {name = self._name}
end 
