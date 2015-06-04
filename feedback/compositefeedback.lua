------------------------------------------------------------------------
--[[ CompositeFeedback ]]--
-- Feedback
-- Composite of many Feedback components
------------------------------------------------------------------------
local CompositeFeedback, parent = torch.class("dp.CompositeFeedback", "dp.Feedback")
CompositeFeedback.isCompositeFeedback = true

function CompositeFeedback:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, feedbacks = xlua.unpack(
      {config},
      'CompositeFeedback', 
      'Composite of many Feedback components',
      {arg='feedbacks', type='table', req=true,
       help='list of feedbacks'}
   )
   self._feedbacks = feedbacks
   config.name = 'compositefeedback'
   parent.__init(self, config)
end

function CompositeFeedback:setup(config)
   parent.setup(self, config)
   for k, v in pairs(self._feedbacks) do
      v:setup(config)
   end
end

function CompositeFeedback:_add(batch, output, report)
   _.map(self._feedbacks, 
      function(key, fb) 
         fb:add(batch, output, report)
      end
   )
end

function CompositeFeedback:report()
   -- merge reports
   local report = {}
   for k, feedback in pairs(self._feedbacks) do
      table.merge(report, feedback:report())
   end
   return report
end

function CompositeFeedback:_reset()
   for k, feedback in pairs(self._feedbacks) do
      feedback:reset()
   end
end

function CompositeFeedback:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
   for k, v in pairs(self._feedbacks) do
      v:verbose(self._verbose)
   end
end
