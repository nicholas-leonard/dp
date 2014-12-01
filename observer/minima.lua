------------------------------------------------------------------------
--[[ Minima ]]--
-- Observer.
-- Notifies listeners when a new minima is found
-- Should only be called on Experiment, Propagator or Model subjects.
-- There should only be once instance of this observer per Experiment.
------------------------------------------------------------------------
local Minima, parent = torch.class("dp.Minima", "dp.Observer")
Minima.isMinima = true

function Minima:__init(config) 
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args
   args, self._start_epoch, self._error_report, self._error_channel, self._maximize
      = xlua.unpack(
      {config},
      'Minima', 
      'Notifies listeners when a new minima is found. ' ..
      'Error can be obtained from experiment report or mediator ' ..
      'channel. If obtained from experiment report via error_report, ' ..
      'the object subscribes to doneEpoch channel.',
      {arg='start_epoch', type='number', default=5,
       help='when to start notifying listeners of new minimas.'},
      {arg='error_report', type='table', 
       help='a sequence of keys to access error from report. ' ..
       'Default is {"validator", "loss", "avgError"}, unless ' ..
       'of course an error_channel is specified.'},
      {arg='error_channel', type='string | table',
       help='channel to subscribe to for early stopping. Should ' ..
       'return an error value for which the models should be ' ..
       'minimized, and the report of the experiment.'},
      {arg='maximize', type='boolean', default=false,
       help='when true, the error channel or report is negated. ' ..
       'This is useful when the channel returns an accuracy ' ..
       'that should be maximized, instead of an error that should not'}
   )
   self._minima_epoch = self._start_epoch - 1
   self._sign = self._maximize and -1 or 1
   assert(not(self._error_report and self._error_channel))
   if not (self._error_report or self._error_channel) then
      self._error_report = {'validator','loss','avgError'}
   end
   parent.__init(self, "doneEpoch")
end

function Minima:setup(config)
   parent.setup(self, config)
   if self._error_channel then
      self._mediator:subscribe(self._error_channel, self, "compareError")
   end
end

function Minima:doneEpoch(report, ...)
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

function Minima:compareError(current_error, ...)
   -- if maximize is true, sign will be -1
   local found_minima = false
   current_error = current_error * self._sign
   if self._epoch >= self._start_epoch then
      if (not self._minima) or (current_error < self._minima) then
         self._minima = current_error
         self._minima_epoch = self._epoch
         found_minima = true
         self._mediator:publish("foundMinima", self)
      end
   end
   return found_minima, current_error
end

function Minima:minima()
   return self._minima, self._minima_epoch
end


