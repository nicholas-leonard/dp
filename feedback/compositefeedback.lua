------------------------------------------------------------------------
--[[ CompositeFeedback ]]--
-- Feedback
-- Composite of many Feedback components
------------------------------------------------------------------------
local CompositeFeedback, parent 
   = torch.class("dp.CompositeFeedback", "dp.Feedback")

function CompositeFeedback:__init(config)
   local args, feedbacks = xlua.unpack(
      'CompositeFeedback', nil,
      {arg='feedbacks', type='table'}
   )
   self._feedbacks = feedbacks
   config.name = 'compositefeedback'
   parent.__init(self, config)
end

function CompositeFeedback:setup(config)
   parent.setup(self, config)
   for k, v in pairs(self._feedbacks) do
      v:setup(self._mediator)
   end
end

function CompositeFeedback:report()
   -- merge reports
   local report = {}
   for k, feedback in pairs(self._feedbacks) do
      merge(report, feedback:report())
   end
   return report
end

function CompositeFeedback:reset()
   for k, feedback in pairs(self._feedbacks) do
      feedback:reset()
   end
end
