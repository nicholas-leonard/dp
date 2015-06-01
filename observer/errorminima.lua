------------------------------------------------------------------------
--[[ ErrorMinima ]]--
-- Observer.
-- Notifies listeners when a new minima is found
-- Should only be called on Experiment, Propagator or Model subjects.
-- It is useful to use the Minima directly, as opposed to using a 
-- subclass like EarlyStopper, when no-such sub-class is present.
------------------------------------------------------------------------
local ErrorMinima, parent = torch.class("dp.ErrorMinima", "dp.Observer")

function ErrorMinima:__init(config) 
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args
   args, self._start_epoch, self._error_report, self._error_channel, 
      self._maximize, self._notify, self._verbose = xlua.unpack(
      {config},
      'ErrorMinima', 
      'Monitors when a new minima over an error variable is found. ' ..
      'Variable can be obtained from experiment report or mediator ' ..
      'channel. If obtained from experiment report via error_report, ' ..
      'the object subscribes to doneEpoch channel.',
      {arg='start_epoch', type='number', default=5,
       help='when to start notifying listeners of new minimas.'},
      {arg='error_report', type='table', 
       help='a sequence of keys to access error from report. ' ..
       'Default is {"validator", "loss"}, unless ' ..
       'of course an error_channel is specified.'},
      {arg='error_channel', type='string | table',
       help='channel to subscribe to for monitoring error. Should ' ..
       'return an error value and the last experiment report.'},
      {arg='maximize', type='boolean', default=false,
       help='when true, the error channel or report is negated. ' ..
       'This is useful when the channel returns an accuracy ' ..
       'that should be maximized, instead of an error that should not'},
      {arg='notify', type='boolean', default=true,
       help='Notifies listeners when a new minima is found.'},
      {arg='verbose', type='boolean', default=true,
       help='provide verbose outputs every epoch'}
   )
   self._minima_epoch = self._start_epoch - 1
   self._sign = self._maximize and -1 or 1
   assert(not(self._error_report and self._error_channel))
   if not (self._error_report or self._error_channel) then
      self._error_report = {'validator','loss'}
   end
   parent.__init(self, "doneEpoch")
end

function ErrorMinima:setup(config)
   parent.setup(self, config)
   if self._error_channel then
      self._mediator:subscribe(self._error_channel, self, "compareError")
   end
end

function ErrorMinima:doneEpoch(report, ...)
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

function ErrorMinima:compareError(current_error, ...)
   -- if maximize is true, sign will be -1
   local found_minima = false
   current_error = current_error * self._sign
   if self._epoch >= self._start_epoch then
      if (not self._minima) or (current_error < self._minima) then
         self._minima = current_error
         self._minima_epoch = self._epoch
         found_minima = true
      end
      
      if self._notify then
         self._mediator:publish("errorMinima", found_minima, self)
      end
   end
   return found_minima, current_error
end

function ErrorMinima:minima()
   return self._minima, self._minima_epoch
end
