------------------------------------------------------------------------
--[[ Perplexity ]]--
-- Feedback
-- Computes perplexity for language models
-- For now, only works with SoftmaxTree
------------------------------------------------------------------------
local Perplexity, parent = torch.class("dp.Perplexity", "dp.Feedback")
Perplexity.isPerplexity = true

function Perplexity:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, name = xlua.unpack(
      {config},
      'Perplexity',
      'Computes perplexity for language models',
      {arg='name', type='string', default='perplexity',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   parent.__init(self, config)
   self._nll = 0
end

function Perplexity:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Perplexity:perplexity()
   -- exponential of the mean NLL
   return math.exp(self._nll / self._n_sample)
end

function Perplexity:doneEpoch(report)
   if self._n_sample > 0 and self._verbose then
      print(self._id:toString().." perplexity = "..self:perplexity())
   end
end

function Perplexity:_add(batch, output, carry, report)
   if output:input():dim() == 2 then
      -- assume output originates from LogSoftMax
      local act = output:forward('bf')
      if act ~= 'torch.DoubleTensor' and act ~= 'torch.FloatTensor' then
         act = output:forward('bf', 'torch.FloatTensor')
      end
      local targets = batch:targets():forward('b')
      local sum = 0
      for i=1,targets:size(1) do
         sum = sum + act[i][targets[i]]
      end

      self._nll = self._nll - sum
   else
      -- assume output originates from SoftMaxTree
      local act = output:forward('b')
      if act ~= 'torch.DoubleTensor' and act ~= 'torch.FloatTensor' then
         act = output:forward('b', 'torch.FloatTensor')
      end
      -- accumulate the sum of negative log likelihoods
      self._nll = self._nll - act:sum()
   end
end

function Perplexity:_reset()
   self._nll = 0
end

function Perplexity:report()
   return {
      [self:name()] = {
         perplexity = self._n_sample > 0 and self:perplexity() or 0
      },
      n_sample = self._n_sample
   }
end
