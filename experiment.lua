require 'torch'

require 'utils'

------------------------------------------------------------------------
--[[ LearningRateSchedule ]]--
------------------------------------------------------------------------

local LearningRateSchedule = torch.class("dp.LearningRateSchedule")

LearningRateSchedule.isEpochObserver = true

function LearningRateSchedule:__init(...)
   local args, schedule = xlua.unpack(
      'LearningRateSchedule', nil,
      {arg='schedule', type='table', req=true,
       help='Epochs as keys, and learning rates as values'}
   )
   self._schedule = schedule
end

function LearningRateSchedule:onDoneEpoch(mediator)
   local epoch = mediator:epoch()
   local learning_rate = self._schedule[epoch]
   if learning_rate then
      mediator:trainer():setLearningRate(learning_rate)
   end
end

------------------------------------------------------------------------
--[[ Mediator ]]--
------------------------------------------------------------------------

local Mediator = torch.class("dp.Mediator")

Mediator.isMediator = true

function Mediator:__init(...)
   self._observers   
   self._epoch_observers 
      = _.where(self._observers, {isEpochObserver=true})
end

function Mediator:doneEpoch(colleague)
   
end

function Mediator:doneExperimentEpoch(colleague)
   for i = 1,#self._experiment_epoch_observers do
      self._experiment_epoch_observers[i]:doneEpoch(self)
   end
end

function Mediator:doneTrainerEpoch(colleague)
   for i = 1,#self._train_epoch_observers do
      self._train_epoch_observers[i]:doneEpoch(self)
   end
end

function Mediator:doneValidatorEpoch(colleague)
   for i = 1,#self._validator_epoch_observers do
      self._valid_epoch_observers[i]:doneEpoch(self)
   end
end


function Mediator:observers()
   return self._observers
end

------------------------------------------------------------------------
--[[ Experiment ]]--
------------------------------------------------------------------------

local Experiment = torch.class("dp.Experiment", "dp.Mediator")

Experiment.isExperiment = true

function Experiment:__init(...)
   local args, trainer, validator, tester, terminator, logger,
      observers, random_seed
      = xlua.unpack(
         {... or {}}
         'Experiment', nil
         {arg='trainer', type='dp.Trainer'}
   )
   Mediator.__init(self, {observers=observers})
end

function Experiment:run()
   while self._terminate:doEpoch() do
      self._trainer:doEpoch()
      self._validator:doEpoch()
   end
end

function Experiment:trainer()
   return self._trainer
end

function Experiment:validator()
   return self._validator
end

function Experiment:tester()
   return self._tester
end

function Experiment:logger()
   return self._logger
end

function Experiment:randomSeed()
   return self._random_seed
end



--[[
local Experiment = torch.class("dp.ExperimentLoader")

function ExperimentLoader:__init(...)
   local args, 
]]--
