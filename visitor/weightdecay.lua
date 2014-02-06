------------------------------------------------------------------------
--[[ WeightDecay ]]--
-- ModelVisitor
-- Decays the weight of the visited parameterized models.
------------------------------------------------------------------------

local WeightDecay, parent = torch.class("dp.WeightDecay", "dp.Visitor")

function WeightDecay:__init(config)
   config = config or {}
   local args, wd_factor, name = xlua.unpack(
      {config},
      'WeightDecay', nil,
      {arg='wd_factor', type='number', help='Weight decay factor'},
      {arg='name', type='string', default='weightdecay',
       help='identifies visitor in reports.'}
   )
   self._wd_factor = wd_factor
   config.name = name
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.exclude = config.exclude or {}
   table.insert(config.exclude, 'no-weightdecay')
   parent.__init(self, config)
end

function WeightDecay:_visitModel(model)
   local params = model:parameters()
   for param_name, param_table in pairs(params) do
      if param_name ~= 'bias' then
         param_table.grad:add(self.wd_factor, param_table.param)
      end
   end
end
