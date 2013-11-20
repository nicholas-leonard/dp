require 'torch'

------------------------------------------------------------------------
--[[ Feedback ]]--
-- Strategy
-- strategies for processing predictions and targets. 
-- Unlike observers, feedback strategies generate reports.
-- Like observers they may also publish/subscribe to mediator channels.
-- When serialized with the model, they may also be unserialized to
-- generate graphical reports.

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
------------------------------------------------------------------------
local Feedback = torch.class("dp.Feedback")
Feedback.isFeedback = true

function Feedback:__init(...)
   local args, criterion = xlua.unpack(
      'Feedback', nil,
      {arg='criterion', type='nn.Criterion'}
   )
   self._criterion = criterion
end

function Feedback:setup(mediator)
   self._mediator = mediator
end

function Feedback:add(predictions, targets)
   local err = self._criterion:forward(predictions, targets)
   local gradCriterion = self._criterion:backward(predictions, targets)
   
end

local MultiFeedback = torch.class("dp.MultiFeedback")

function MultiFeedback:__init(...)
   local args, feedbacks = xlua.unpack(
      'MultiFeedback', nil,
      {arg='feedbacks', type='table'}
   )
   self._feedbacks = feedbacks
end

function MultiFeedback:setup(mediator)
   for 
end


