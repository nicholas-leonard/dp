------------------------------------------------------------------------
--[[ VisitorChain ]]--
-- Composite, Visitor, Chain of Responsibility
------------------------------------------------------------------------
local VisitorChain, parent = torch.class("dp.VisitorChain", "dp.Visitor")

function VisitorChain:__init(config)
   local args, visitors = xlua.unpack(
      {config},
      'VisitorChain', nil,
      {arg='visitors', type='table', req=true}
   )
   config.name = 'visitorchain'
   parent.__init(self, config)
   self._visitors = visitors
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
      merge(report, visitor:report())
   end
   return report
end
