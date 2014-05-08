------------------------------------------------------------------------
--[[ TreeNLL ]]--
-- Loss subclass
-- Negative Log Likelihood for SoftmaxTrees.
-- Used for maximizing the likelihood of SoftmaxTree Model outputs.
-- SoftmaxTree outputs a column tensor representing the likelihood
-- of each target in the batch. Thus SoftmaxTree requires the targets.
-- So this Loss only computes the negative log of those outputs, as 
-- well as its corresponding gradients.
------------------------------------------------------------------------
local TreeNLL, parent = torch.class("dp.TreeNLL", "dp.Loss")
TreeNLL.isTreeNLL = true

function TreeNLL:__init(config)
   self._module = nn.Sequential()
   self._module:add(nn.Log())
   self._module:add(nn.Mean())
   config = config or {}
   parent.__init(self, config)
   self._output_grad = torch.Tensor{-1}
end

function TreeNLL:_forward(carry)
   local input = self:inputAct()
   self.loss = -self._module:forward(input)[1]
   return carry
end

function TreeNLL:_backward(carry)
   local input = self:inputAct()
   self:inputGrad(self._module:backward(input, self._output_grad))
   return carry
end

function TreeNLL:_type(type)
   self._input_type = type
   -- this actually doesn't change anything
   self._output_type = type 
   self._module:type(type)
   self._output_grad = self._output_grad:type(type)
end
