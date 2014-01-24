------------------------------------------------------------------------
--[[ LearningRateSchedule ]]--
-- Observer
-- Only works on dp.Learn subject
------------------------------------------------------------------------
local LearningRateSchedule, parent
   = torch.class("dp.LearningRateSchedule", "dp.Observer")

function LearningRateSchedule:__init(...)
   local args, schedule = xlua.unpack(
      {... or {}},
      'LearningRateSchedule', nil,
      {arg='schedule', type='table | tensor', req=true,
       help='Epochs as keys, and learning rates as values'}
   )
   self._schedule = schedule
   parent.__init(self, "doneEpoch")
end

function LearningRateSchedule:setSubject(subject)
   assert(subject.isLearn)
   self._subject = subject
end

function LearningRateSchedule:doneEpoch(report, ...)
   assert(type(report) == 'table')
   local learning_rate = self._schedule[report.epoch]
   if learning_rate then
      self._subject:setLearningRate(learning_rate)
   end
end
