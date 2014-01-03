------------------------------------------------------------------------
--[[ EarlyStopper ]]--
-- Observer.
-- Saves version of the subject with the lowest error and terminates
-- the experiment when no new minima is found for max_epochs.
-- Should only be called on Experiment, Propagator or Model subjects.
------------------------------------------------------------------------

local EarlyStopper, parent = torch.class("dp.EarlyStopper", "dp.Observer")
EarlyStopper.isEarlyStopper = true

function EarlyStopper:__init(config) 
   config = config or {}
   local args, start_epoch, error_report, error_channel, maximize, 
         save_strategy, max_epochs, max_error, min_epoch
      = xlua.unpack(
      {config},
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
      {arg='maximize', type='boolean', default=false,
       help='when true, the error channel or report is negated. ' ..
       'This is useful when the channel returns an accuracy ' ..
       'that should be maximized, instead of an error that should not'},
      {arg='save_strategy', type='object', default=dp.SaveToFile(),
       help='a serializable object that has a :save(subject) method.'},
      {arg='max_epochs', type='number', default='30',
       help='maximum number of epochs to consider after a minima ' ..
       'has been found. After that, a terminate signal is published ' ..
       'to the mediator.'},
      {arg='max_error', type='number', 
       help='maximum value for which the experiment should be ' ..
       'stopped after min_epochs. ' ..
       'If maximize is true, this is min value'},
      {arg='min_epoch', type='number', default=10000000,
       help='see max_value'}
   )
   self._start_epoch = start_epoch
   self._minima_epoch = start_epoch - 1
   self._error_report = error_report
   self._error_channel = error_channel
   self._save_strategy = save_strategy
   self._maximize = maximize
   self._min_epoch = min_epoch
   self._sign = 1
   if maximize then
      self._sign = -1
   end
   self._max_epochs = max_epochs
   self._max_error = max_error * self._sign
   assert(self._error_report or self._error_channel)
   assert(not(self._error_report and self._error_channel))
   if not (self._error_report or self._error_channel) then
      self._error_report = {'validator','loss'}
   end
   parent.__init(self, "doneEpoch")
end

function EarlyStopper:setSubject(subject)
   assert(subject.isModel 
      or subject.isPropagator 
      or subject.isExperiment)
   self._subject = subject
end

function EarlyStopper:setup(config)
   parent.setup(self, config)
   if self._error_channel then
      self._mediator:subscribe(self._error_channel, self, "compareError")
   end
   self._save_strategy:setup(self._subject)
end

function EarlyStopper:doneEpoch(report, ...)
   assert(type(report) == 'table')
   self._epoch = report.epoch
   if self._error_report then
      local report_cursor = report
      for _, name in ipairs(self._error_report) do
         report_cursor = report_cursor[name]
      end
      self:compareError(report_cursor, ...)
   end
end

function EarlyStopper:compareError(current_error, ...)
   -- if maximize is true, sign will be -1
   local found_minima = false
   current_error = current_error * self._sign
   if self._epoch >= self._min_epoch then
      if current_error > self._max_error then
         self._mediator:publish("doneExperiment")
      end
   end
   if self._epoch >= self._start_epoch then
      if (not self._minima) or (current_error < self._minima) then
         self._minima = current_error
         self._minima_epoch = self._epoch
         self._save_strategy:save(self._subject, current_error)
         found_minima = true
      end
   end
   if self._max_epochs < (self._epoch - self._minima_epoch) then
      print("found minima : " .. self._minima .. 
            " at epoch " .. self._minima_epoch) 
      self._mediator:publish("doneExperiment")
   end
   return found_minima
end
