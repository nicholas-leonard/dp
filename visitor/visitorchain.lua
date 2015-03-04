------------------------------------------------------------------------
--[[ VisitorChain ]]--
-- Composite, Visitor, Chain of Responsibility
------------------------------------------------------------------------
local VisitorChain, parent = torch.class("dp.VisitorChain", "dp.Visitor")
VisitorChain.isVisitorChain = true

function VisitorChain:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, visitors = xlua.unpack(
      {config},
      'VisitorChain', 
      'A composite chain of visitors. The order is important.',
      {arg='visitors', type='table', req=true,
       help='sequence of visitors to apply to visited models'}
   )
   config.name = 'visitorchain'
   parent.__init(self, config)
   self._visitors = visitors
   -- the root visitor will perform the zeroGradParameters on models
   local zero_grads = self._zero_grads
   self:setZeroGrads(false)
   self._zero_grads = zero_grads
end

function VisitorChain:setup(config)
   parent.setup(self, config)
   for i, visitor in ipairs(self._visitors) do
      visitor:setup(config)
   end
end

function VisitorChain:_visitModel(model)
   for i, visitor in ipairs(self._visitors) do
      visitor:visitModel(model)
   end
end

function VisitorChain:report()
   -- merge reports
   local report = {}
   for k, visitor in pairs(self._visitors) do
      table.merge(report, visitor:report())
   end
   return report
end

function VisitorChain:setZeroGrads(zero_grads)
   for k, visitor in pairs(self._visitors) do
      visitor:setZeroGrads(zero_grads)
   end
   self._zero_grads = zero_grads
end

function VisitorChain:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
   for k, v in pairs(self._visitors) do
      v:verbose(self._verbose)
   end
end
