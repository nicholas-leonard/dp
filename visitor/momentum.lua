------------------------------------------------------------------------
--[[ Momentum ]]--
-- Visitor
-- Applies momentum to parameters
------------------------------------------------------------------------
local Momentum, parent = torch.class("dp.Momentum", "dp.Visitor")
Momentum.isMomentum = true

function Momentum:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, momentum_factor, damping_factor, nesterov, name
      = xlua.unpack(
      {config},
      'Momentum', 
      'Applies momentum to parameters',
      {arg='momentum_factor', type='number', req=true},
      {arg='damping_factor', type='number between 0 and 1', 
       help='Reduces oscillations. Mostly useful for recurrent nets'},
      {arg='nesterov', type='boolean', default=false},
      {arg='name', type='string', default='momentum',
       help='identifies visitor in reports.'}
   )
   --Damping is an influence within or upon an oscillatory system that 
   --has the effect of reducing, restricting or preventing its 
   --oscillations. In physical systems, damping is produced by 
   --processes that dissipate the energy stored in the oscillation
   self._momentum_factor = momentum_factor
   self._damping_factor = damping_factor or momentum_factor
   self._nesterov = nesterov
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.exclude = config.exclude or {}
   table.insert(config.exclude, 'no-momentum')
   table.insert(config.exclude, 'accUpdate')
   config.name = name
   parent.__init(self, config)
end

function Momentum:_visitModel(model)
   model:updateGradParameters(self._momentum_factor, self._damping_factor, self._nesterov)
end
