------------------------------------------------------------------------
--[[ Model ]]--
-- Node subclass
-- Adapter of nn.Modules
------------------------------------------------------------------------
local Model, parent = torch.class("dp.Model", "dp.Node")
Model.isModel = true

function Model:__init(...)
   local args, typename, tags, mvstate = xlua.unpack(
      {... or {}},
      'Model', nil,
      {arg='typename', type='string', req=true, 
       help='identifies Model type in reports.'},
      {arg='tags', type='table', default={},
       help='table of tags used for determining which visitors ' ..
       'are allowed to visit the model'},
      {arg='mvstate', type='table', default={},
       help='model-visitor state'}
   )
   self._typename = typename
   self._tags = tags -- tags
   self._report = {} -- stores a report
   self.mvstate = mvstate -- stores stuff for visitors in between passes
   self:doneBatch()
end

function Model:setup(config)
   local args, container
      = xlua.unpack(
      {config or {}},
      'Model:setup', nil,
      {arg='container', type='dp.Container'}
   )
   -- context
   self._container = container
   parent.setup(self, config)
end

function Model:tags()
   return self._tags
end

function Model:parameters()
   return {}
end

function Model:forward(input, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor input")
   self.input.act = input
   self:updateStatistics(carry)
   carry = self:_forward(carry) or carry
   assert(self.output.act.isBaseTensor, "Expecting dp.BaseTensor output")
   self.forwarded = true
   return self.output.act, carry
end

function Model:evaluate(input, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor instance")
   self.input.act = input
   carry.evaluate = true
   self:updateStatistics(carry)
   carry = self:_evaluate(carry) or carry
   assert(self.output.act.isBaseTensor, "Expecting dp.BaseTensor output")
   self.evaluated = true
   self.forwarded = true
   return self.output.act, carry
end

function Model:accept(visitor)
   self.visited = true
   self:_accept(visitor)
end

function Model:_accept(visitor)
   return visitor:visitModel(self)
end

function Model:doneBatch(...)
   parent.doneBatch(self, ...)
   if self.backwarded then
      self:zeroGradParameters()
   end
   self.visited = false
   self.output = {}
end

function Model:_doneBatch(...)
end

function Model:zeroGradParameters()
   for param_name, param_table in pairs(self:parameters()) do
      param_table.grad:zero()
   end
end

function Model:reset()
   error"Not Implemented"
end
