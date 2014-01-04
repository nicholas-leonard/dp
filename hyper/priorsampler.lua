------------------------------------------------------------------------
--[[ PriorSampler ]]--
-- Factory
-- Samples hyper-parameters from a user-defined prior 
-- to initialize and run an experiment (see hyperoptimize.lua example)
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
   print(hyperparams)
   return hyperparams
end
