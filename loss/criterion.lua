------------------------------------------------------------------------
--[[ Criterion ]]--
-- Loss subclass
-- General Adapter of nn.Criterion
------------------------------------------------------------------------
local Criterion, parent = torch.class("dp.Criterion", "dp.Loss")
Criterion.isCriterion = true

function Criterion:__init(config)
   config = config or {}
   local args, criterion, input_view = xlua.unpack(
      {config},
      'Criterion', 
      'Decorates/Adapts a nn.Criterion to the dp.Loss interface',
      {arg='criterion', type='nn.Criterion', req=true,
       help='nn.Criterion that will be adapted to dp.Loss'},
      {arg='input_view', type='string', default='default',
       help='view of the input like "bt", "b", etc.'}
   )
   self._criterion = criterion
   config.input_view = input_view
   parent.__init(self, config)
end
