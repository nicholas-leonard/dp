------------------------------------------------------------------------
--[[ NLL ]]--
-- Loss subclass
-- Adapter of nn.ClassNLLCriterion
-- Negative Log Likelihood 
------------------------------------------------------------------------
local NLL, parent = torch.class("dp.NLL", "dp.Loss")
NLL.isNLL = true

function NLL:__init(config)
   self._criterion = nn.ClassNLLCriterion()
   config = config or {}
   config.target_type = config.target_type or 'torch.IntTensor'
   parent.__init(self, config)
end

function NLL:_forward(carry)
   local input = self:inputAct()
   local target = self.input.target:class(self._output_type)
   self.loss = self._criterion:forward(input, target)
   return carry
end

function NLL:_backward(carry)
   local input = self:inputAct()
   local target = self.input.target:class(self._output_type)
   self:inputGrad(self._criterion:backward(input, target))
   return carry
end

function NLL:_type(type)
   if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' then
      self._input_type = type
   elseif type == 'torch.IntTensor' or type == 'torch.LongTensor' then
      self._output_type = type
   end
end
