require 'torch'

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

------------------------------------------------------------------------
--[[ Criteria ]]--
-- Feedback
-- Adapter that feeds back and accumulates the error of one or many
-- nn.Criterion. Each supplied nn.Criterion requires a name for 
-- reporting purposes. Default name is typename minus module name(s)
------------------------------------------------------------------------
local Criteria, parent 
   = torch.class("dp.Criteria", "dp.Feedback")


function Criteria:__init(config)
   local args, criteria, name, typename_pattern
      = xlua.unpack(
      {config},
      'Criteria', nil,
      {arg='criteria', type='nn.Criterion | table', req=true,
       help='list of criteria to monitor'},
      {arg='name', type='string', default='criteria'},
      {arg='typename_pattern', type='string', 
       help='require criteria to have a torch.typename that ' ..
       'matches this pattern', default="^nn[.]%a*Criterion$"}
   )
   config.name = name
   parent.__init(self, config)
   
   self._criteria = {}
   self._name = name
   if torch.typename(criteria) then
      criteria = {criteria}
   end
   
   for k,v in pairs(criteria) do
      -- non-list items only
      if type(k) ~= 'number' then
         self._criteria[k] = v
      end
   end
   
   for i,v in ipairs(criteria) do
      -- for listed criteria, default name is derived from typename
      local k = _.split(torch.typename(criteria), '.')
      k = k[#k]
      -- remove suffix 'Criterion'
      if string.sub(k, -9) == 'Criterion' then
         k = string.sub(k, 1, -10)
      end
      -- make lowercase
      k = string.lower(k)
      self._criteria[k] = v
   end
   if typepattern ~= '' then
      for k,v in pairs(self._criteria) do
         assert(typepattern(v,typepattern), "Invalid criteria typename")
      end
   end
   
   self._errors = {}
   self:reset()
end

function Criteria:reset()
   -- reset error sums to zero
   for k,v in self._criteria do
      self._errors[k] = 0
   end
   self._samples_seen = 0
end

function Criteria:add(batch)             
   local current_error
   for k,v in self._criteria do
      current_error = v:forward(batch:outputs(), batch:targets())
      self._errors[k] =  (
                              ( self._samples_seen * self._errors[k] ) 
                              + 
                              ( batch:nSample() * current_error )
                         ) 
                         / 
                         self._samples_seen + batch:nSample()
      --TODO gather statistics on backward outputGradients?
   end
   self._samples_seen = self._samples_seen + batch:nSample()
end

function Criteria:report()
   return { 
      [self:name()] = self._errors,
      n_sample = self._samples_seen
   }
end


