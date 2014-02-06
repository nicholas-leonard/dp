------------------------------------------------------------------------
--[[ Momentum ]]--
-- ModelVisitor
-- Applies momentum to parameters
------------------------------------------------------------------------

local Momentum, parent = torch.class("dp.Momentum", "dp.Visitor")

function Momentum:__init(config)
   config = config or {}
   local args, momentum_factor, damping_factor, nesterov, name
      = xlua.unpack(
      {config},
      'Momentum', 'Applies momentum to parameters',
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
   config.name = name
   parent.__init(self, config)
end

function Momentum:_visitModel(model)
   if self._momentum_factor == 0 then 
      return 
   end
   local params = model:parameters()
   for param_name, param_table in pairs(params) do
      if not param_table.past_grad then
         param_table.past_grad 
            = torch.Tensor():typeAs(
                  param_table.grad
               ):resizeAs(
                  param_table.grad
               ):copy(
                  param_table.grad
               )
      else
         param_table.past_grad:mul(
               self._momentum_factor
            ):add(
               1-self._damping_factor, param_table.grad
            )
      end
      if self._nesterov then
         param_table.grad:add(
               self._momentum_factor, param_table.past_grad
            )
      else
         param_table.grad = param_table.past_grad
      end
   end
end
