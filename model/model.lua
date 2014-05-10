------------------------------------------------------------------------
--[[ Model ]]--
-- Node subclass
-- Adapter of nn.Modules
------------------------------------------------------------------------
local Model, parent = torch.class("dp.Model", "dp.Node")
Model.isModel = true

function Model:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, typename, tags, mvstate = xlua.unpack(
      {config},
      'Model', nil,
      {arg='typename', type='string', req=true, 
       help='identifies Model type in reports.'},
      {arg='tags', type='table', default={},
       help='table of tags used for determining which visitors ' ..
       'are allowed to visit the model'},
      {arg='mvstate', type='table', default={},
       help='model-visitor state. Can be used to specify arguments '..
       'to visitors that will adapt these to the model.'}
   )
   self._typename = typename
   self._tags = tags -- tags
   self._report = {} -- stores a report
   self.mvstate = mvstate -- stores stuff for visitors in between passes
   parent.__init(self, config)
end

function Model:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, container = xlua.unpack(
      {config},
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
   error"Not Implemented"
end

function Model:forward(input, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor input")
   self.input.act = input:shallowClone()
   self:updateStatistics(carry)
   carry = self:_forward(carry) or carry
   assert(self.output.act.isBaseTensor, "Expecting dp.BaseTensor output")
   self.forwarded = true
   return self.output.act, carry
end

function Model:evaluate(input, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor instance")
   self.input.act = input:shallowClone()
   carry.evaluate = true
   self:updateStatistics(carry)
   carry = self:_evaluate(carry) or carry
   assert(self.output.act.isBaseTensor, "Expecting dp.BaseTensor output")
   self.evaluated = true
   self.forwarded = true
   return self.output.act, carry
end

function Model:backward(output, carry)
   assert(output.isBaseTensor, "Expecting dp.BaseTensor output")
   self.output.grad = output:shallowClone()
   carry = self:_backward(carry) or carry
   assert(self.input.grad.isBaseTensor, "Expecting dp.BaseTensor grad")
   self.backwarded = true
   return self.input.grad, carry
end

function Model:inputAct(input_act)
   error"Not Implemented"
end

function Model:inputGrad(input_grad)
   error"Not Implemented"
end

function Model:outputAct(output_act)
   error"Not Implemented"
end

function Model:outputGrad(output_grad)
   error"Not Implemented"
end

function Model:accept(visitor)
   self.visited = true
   self:_accept(visitor)
end

function Model:_accept(visitor)
   return visitor:visitModel(self)
end

function Model:doneBatch(...)
   if self.backwarded then
      self:zeroGradParameters()
   end
   parent.doneBatch(self, ...)
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
