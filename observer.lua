require 'torch'

require 'utils'

--[[ TODO ]]--
--Observer of channels mapped to function names?

------------------------------------------------------------------------
--[[ Observer ]]--
-- An object that is called when events occur.
-- Based on the Subject-Objserver design pattern. 
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
function Observer:setup(mediator, subject, ...)
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


------------------------------------------------------------------------
--[[ MultiObserver ]]--
-- Is composed of multible observers
------------------------------------------------------------------------

local MultiObserver = torch.class("dp.MultiObserver", "dp.Observer")

function MultiObserver:__init(observers)
   self._observers = observers
   for name, observer in pairs(self._observers) do
      assert(observer.isObserver)
   end
   for _, observer in ipairs(self._observers) do
      assert(observer.isObserver)
   end
end

function MultiObserver:setup(mediator, subject, ...)
   self._mediator = mediator
   self:setSubject(subject)
   for name, observer in pairs(self._observers) do
      observer:setup(mediator, subject, ...)
   end
   for _, observer in ipairs(self._observers) do
      observer:setup(mediator, subject, ...)
   end
end

function MultiObserver:report()
   error"NotSupported : observers don't generate reports"
   --[[local report = {}
   for name, observer in pairs(self._observers) do
      assert(observers)
      local observer_report = observer:report()
      if observer_report and not table.eq(observer_report, {})  then
         report[name] = observer_report
      end
   end
   return report]]--
end

------------------------------------------------------------------------
--[[ LearningRateSchedule ]]--
-- Optimizer Observer
-- Can be called from Propagator or Experiment
------------------------------------------------------------------------

local LearningRateSchedule, parent
   = torch.class("dp.LearningRateSchedule", "dp.Observer")

function LearningRateSchedule:__init(...)
   local args, schedule = xlua.unpack(
      'LearningRateSchedule', nil,
      {arg='schedule', type='table', req=true,
       help='Epochs as keys, and learning rates as values'}
   )
   self._schedule = schedule
   parent.__init(self, "doneEpoch")
end

function LearningRateSchedule:setSubject(subject)
   assert(subject.isOptimizer)
   self._subject = subject
end

function LearningRateSchedule:doneEpoch(report, ...)
   assert(type(report) == 'table')
   local learning_rate = self._schedule[report.epoch]
   if learning_rate then
      self._subject:setLearningRate(learning_rate)
   end
end



------------------------------------------------------------------------
--[[ SaveToFile ]]--
-- Strategy.
-- Saves version of the subject with the lowest error
------------------------------------------------------------------------

local SaveToFile = torch.class("dp.SaveToFile")

function SaveToFile:__init(save_dir)
   self._save_dir = save_dir or dp.SAVE_DIR
end

function SaveToFile:save(subject)
   --concatenate save directory with subject id
   filename = paths.concat(self._save_dir, 
                           subject:id():toPath() .. '.dat')
   --creates directories if required
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('SaveToFile: saving to '.. filename)
   --saves subject to file
   torch.save(filename, subject)
end


------------------------------------------------------------------------
--[[ EarlyStopper ]]--
-- Observer.
-- Saves version of the subject with the lowest error and terminates
-- the experiment when no new minima is found for max_epochs.
-- Should only be called on Experiment, Propagator or Model subjects.
------------------------------------------------------------------------

local EarlyStopper, parent 
   = torch.class("dp.EarlyStopper", "dp.Observer")

function EarlyStopper:__init(config) 
   local args, start_epoch, error_report, error_channel, save_strategy,
         max_epochs 
      = xlua.unpack(
      {config or {}},
      'EarlyStopper', 
      'Saves a model at each new minima of error. ' ..
      'Error can be obtained from experiment report or mediator ' ..
      'channel. If obtained from experiment report via error_func, ' ..
      'subscribes to onDoneEpoch channel.',
      {arg='start_epoch', type='number', default=5,
       help='when to start saving models.'},
      {arg='error_report', type='table', 
       help='a sequence of keys to access error from report. ' ..
       "Default is {'validator', 'error'}, unless " ..
       'of course an error_channel is specified.'},
      {arg='error_channel', type='string | table',
       help='channel to subscribe to for early stopping. Should ' ..
       'return an error for which the models should be minimized, ' ..
       'and the report of the experiment.'},
      {arg='save_strategy', type='object', default=SaveToFile(),
       help='a serializable object that has a :save(subject) method.'},
      {arg='max_epochs', type='number', default='30',
       help='maximum number of epochs to consider after a minima ' ..
       'has been found. After that, a terminate signal is published ' ..
       'to the mediator.'}
   )
   self._start_epoch = start_epoch
   self._minima_epoch = start_epoch - 1
   self._error_report = error_report
   self._error_channel = error_channel
   self._save_strategy = save_strategy
   self._max_epochs = max_epochs
   assert(self._error_report or self._error_channel)
   assert(not(self._error_report and self._error_channel)
   if not (self._error_report or self._error_channel) then
      self._error_report = {'validator','loss'}
   end
   parent.__init(self, "doneEpoch")
end

function EarlyStopper:setSubject(subject)
   assert(subject.isModel)
end

function EarlyStopper:setup(mediator, subject, ...)
   Observer.setup(self, mediator, subject, ...)
   if self._error_channel then
      mediator:subscribe(self._error_channel, 
         function(current_error, report, ...)
            self:compareError(current_error, report, ...)
         end
   end
end

function EarlyStopper:doneEpoch(report, ...)
   assert(type(report) == 'table')
   if self._error_report then
      local report_cursor = report
      for _, name in ipairs(self._error_report) do
         report_cursor = report_cursor[name]
      end
      local current_error = report_cursor
      self:compareError(current_error, report, ...)
   end
end

function EarlyStopper:compareError(current_error, report)
   assert(type(report) == 'table')
   local current_epoch = report.epoch
   assert(type(current_epoch) == 'number')
   if current_epoch >= start_epoch then
      if (not self._minima) or (current_error > self._minima) then
         self._minima = current_error
         self._minima_epoch = current_epoch
         self._save_strategy:save(subject)
      end
   end
   if self._max_epochs >= (current_epoch - self._minima_epoch) then
      self._mediator:publish("doneExperiment")
   end
end

