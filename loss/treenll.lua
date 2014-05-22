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
   config.target_type = config.target_type or 'torch.IntTensor'
   config.target_view = 'b'
   config.input_view = 'bf' -- nn.Mean() will fail with b
   parent.__init(self, config)
   self._output_grad = torch.Tensor{-1}
end

function TreeNLL:_forward(carry)
   self.loss = -self._module:forward(self:inputAct())[1]
   return carry
end

function TreeNLL:_backward(carry)
   self:inputGrad(self._module:backward(self:inputAct(), self._output_grad))
   return carry
end

function TreeNLL:_type(type)
   if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' then
      self._input_type = type
      self._module:type(type)
   elseif type == 'torch.IntTensor' or type == 'torch.LongTensor' then
      -- this actually doesn't change anything:
      self._output_type = type
   end
end
