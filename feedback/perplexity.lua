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
      {arg='name', type='string', default='perplexity'}
   )
   config.name = name
   parent.__init(self, config)
   self._perplexity = 0
end

function Perplexity:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Perplexity:perplexity()
   -- exponential of the mean NLL
   return 10^(self._perplexity / (self._n_sample*math.log(10)))
end

function Perplexity:doneEpoch(report)
   if self._n_sample > 0 then
      print(self._id:toString().." perplexity = "..self:perplexity())
   end
end

function Perplexity:_add(batch, output, carry, report)
   local act = output.act:feature()
   if act ~= 'torch.DoubleTensor' and act ~= 'torch.FloatTensor' then
      act = output.act:feature('torch.FloatTensor')
   end
   -- accumulate the sum of negative log likelihoods
   self._perplexity = self._perplexity - act:sum()
end

function Perplexity:_reset()
   self._perplexity = 0
end

function Perplexity:report()
   return { 
      [self:name()] = {
         perplexity = self._n_sample > 0 and self:perplexity() or 0
      },
      n_sample = self._n_sample
   }
end

