------------------------------------------------------------------------
--[[ Observer ]]--
-- An object that is called when events occur.
-- Based on the Subject-Observer design pattern. 
-- Uses a mediator to publish/subscribe to channels.
-- Observers cannot publish reports (for now). 

-- The reason for this is 
-- that a report is required of observers to do their job, thus making
-- it impossible for them to participate to report generation.
-- The only possibility would be to allow for Observers to modify the 
-- received report, but then the ordering of these modifications would
-- be undefined, unless they make use of Mediator priorities.
------------------------------------------------------------------------
local Observer = torch.class("dp.Observer")
Observer.isObserver = true

function Observer:__init(channels, callbacks)
   if type(channels) == 'string' then
      channels = {channels}
   end
   if type(callbacks) == 'string' then
      callbacks = {callbacks}
   end
   self._channels = channels or {}
   self._callbacks = callbacks or channels
end

function Observer:subscribe(channel, callback)
   self._mediator:subscribe(channel, self, callback or channel)
end

--should be reimplemented to validate subject
function Observer:setSubject(subject)
   --assert subject.isSubjectType
   self._subject = subject
end

--An observer is setup with a mediator and a subject.
--The subject is usually the object from which the observer is setup.
function Observer:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, subject = xlua.unpack(
      {config},
      'Observer:setup', nil,
      {arg='mediator', type='dp.Mediator', req=true},
      {arg='subject', type='dp.Experiment | dp.Propagator | ...',
      help='object being observed.'}
   )
   assert(mediator.isMediator)
   self._mediator = mediator
   self:setSubject(subject)
   for i=1,#self._channels do
      self:subscribe(self._channels[i], self._callbacks[i])
   end
end

function Observer:report()
   error"NotSupported : observers don't generate reports"
end

function Observer:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function Observer:silent()
   self:verbose(false)
end
