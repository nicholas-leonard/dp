------------------------------------------------------------------------
--[[ TreeNLL ]]--
-- Loss subclass
-- Negative Log Likelihood for SoftmaxTrees.
-- Used for maximizing the likelihood of SoftmaxTree Model outputs.
-- SoftmaxTree outputs a column tensor representing the log likelihood
-- of each target in the batch. Thus SoftmaxTree requires the targets.
-- So this Loss only computes the negative of those outputs, as 
-- well as its corresponding gradients.
------------------------------------------------------------------------
local TreeNLL, parent = torch.class("dp.TreeNLL", "dp.Loss")
TreeNLL.isTreeNLL = true

function TreeNLL:__init(config)
   self._module = nn.Sequential()
   self._module:add(nn.Mean()) --not in cunn
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
   local input_grad = self._module:backward(input, self._output_grad)
   self:inputGrad(input_grad)
   return carry
end

function TreeNLL:_type(type)
   if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' then
      self._input_type = type
   elseif type == 'torch.IntTensor' or type == 'torch.LongTensor' then
      -- this actually doesn't change anything:
      self._output_type = type
   end
end
