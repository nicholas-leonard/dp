------------------------------------------------------------------------
--[[ Feedback ]]--
-- Strategy
-- strategies for processing predictions and targets. 
-- Unlike observers, feedback strategies generate reports.
-- Like observers they may also publish/subscribe to mediator channels.
-- When serialized with the model, they may also be unserialized to
-- generate graphical reports (see Confusion).
------------------------------------------------------------------------
local Feedback = torch.class("dp.Feedback")
Feedback.isFeedback = true

function Feedback:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, verbose, name = xlua.unpack(
      {config},
      'Feedback', 
      'strategies for processing predictions and targets.',
      {arg='verbose', type='boolean', default=true,
       help='provide verbose outputs every epoch'},
      {arg='name', type='string', req=true,
       help='used to identify report'}
   )
   self._name = name
   self._verbose = verbose
   self._n_sample = 0
end

function Feedback:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, propagator, dataset = xlua.unpack(
      {config},
      'Feedback:setup', 
      'setup the Feedback for mediation and such',
      {arg='mediator', type='dp.Mediator', 
       help='used for inter-object communication. defaults to dp.Mediator()'},
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
function Feedback:add(batch, output, carry, report)
   assert(batch.isBatch, "First argument should be Batch")
   self._n_sample = self._n_sample + batch:nSample()
   self:_add(batch, output, carry, report)
end

function Feedback:_add(batch, output, carry, report)
end

function Feedback:report()
   return {}
end

function Feedback:reset()
   self._n_sample = 0
   self:_reset()
end

function Feedback:_reset()
end

function Feedback:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function Feedback:silent()
   self:verbose(false)
end
