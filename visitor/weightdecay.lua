------------------------------------------------------------------------
--[[ WeightDecay ]]--
-- ModelVisitor
-- Decays the weight of the visited parameterized models.
------------------------------------------------------------------------
local WeightDecay, parent = torch.class("dp.WeightDecay", "dp.Visitor")
WeightDecay.isWeightDecay = true

function WeightDecay:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, wd_factor, name = xlua.unpack(
      {config},
      'WeightDecay', 
      'Decays the weight of the visited parameterized models.',
      {arg='wd_factor', type='number', req=true, 
       help='Weight decay factor'},
      {arg='name', type='string', default='weightdecay',
       help='identifies visitor in reports.'}
   )
   self._wd_factor = wd_factor
   config.name = name
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.exclude = config.exclude or {}
   table.insert(config.exclude, 'no-weightdecay')
   table.insert(config.exclude, 'accUpdate')
   parent.__init(self, config)
end

function WeightDecay:_visitModel(model)
   local params, gradParams = model:parameters()
   for k, param in pairs(params) do
      gradParams[k]:add(self._wd_factor, param)
   end
end
