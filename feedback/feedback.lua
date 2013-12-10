------------------------------------------------------------------------
--[[ Feedback ]]--
-- Strategy
-- strategies for processing predictions and targets. 
-- Unlike observers, feedback strategies generate reports.
-- Like observers they may also publish/subscribe to mediator channels.
-- When serialized with the model, they may also be unserialized to
-- generate graphical reports (see Confusion).

-- Discussion :
-- Need a way to inform components of error for various feedback units
-- For logging, report will be enough, but for samplers, models and such
-- that require access to the error of individual examples for every 
-- epoch, its not enough. 
-- First solution is to allow for feedback to modify batch object that 
-- will be returned during backwards pass. Second solution would be 
-- to use the mediator : have feedback publish individual errors, 
-- have model subscribe to this channel. The second solution is much 
-- more flexible, in that it doesn't require the Batch object to have 
-- preimplemented methods. Furthermore, we could save ourselves the 
-- hasle of reimplementing the criteria, by making the Feedback a 
-- propagator constructor parameter. It receives outputs, targets, 
-- can measure error and send it back. If this proves insufficiently
-- flexible

-- TODO :
-- Feedbacks publish feedback to channels which visitors, models, etc 
--    can subscribe to
------------------------------------------------------------------------
local Feedback = torch.class("dp.Feedback")
Feedback.isFeedback = true

function Feedback:__init(...)
   local args, name = xlua.unpack(
      {... or {}},
      'Feedback', nil,
      {arg='name', type='string', req=true,
       help='used to identify report'}
   )
   self._name = name
   self._samples_seen = 0
end

function Feedback:setup(...)
   local args, mediator, propagator, dataset = xlua.unpack(
      {... or {}},
      'Feedback:setup', nil,
      {arg='mediator', type='dp.Mediator'},
      {arg='propagator', type='dp.Propagator'},
      {arg='dataset', type='dp.DataSet', 
       help='This might be useful to determine the type of targets. ' ..
       'Feedback should not hold a reference to the dataset due to ' ..
       "the feedback's possible serialization."}
   )
   self._mediator = mediator
   self._propagator = propagator
   if self._name then
      self._id = propagator:id():create(self._name)
   end
   self._name = nil
   return dataset
end

function Feedback:id()
   return self._id
end

function Feedback:name()
   return self._id:name()
end

--accumulates information from the batch
function Feedback:add(batch, report)
   assert(batch.isBatch)
   self._samples_seen = self._samples_seen + batch:nSample()
end

function Feedback:report()
   return {}
end

function Feedback:reset()
   self._samples_seen = 0
end
