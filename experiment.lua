require 'torch'
require 'optim' --for logger

require 'utils'

------------------------------------------------------------------------
--[[ Observer ]]--
-- An object that is called when events occur.
-- Supported events are :
-- doneBatch
-- doneEpoch
-- doneAll

-- experiment:doneAll(caller, experiment)
-- experiment:doneEpoch(caller, experiment)

-- propagator:doneAll(caller, experiment)
-- propagator:doneEpoch(caller, experiment)
-- propagator:doneBatch(caller, experiment)

-- Possible callers are:
-- propagator
-- experiment
-- etc.
--
-- This isn't exactly the Subject-Objserver design pattern.
-- By passing the experiment and the caller, we allow both to be 
-- subjects, in a sense. The reason for passing the experiment is to 
-- provide the observer with access to all methods of the experiment,
-- which includes optimizer, validator, tester, epoch, and so on.
-- This allows for great flexibility. 
------------------------------------------------------------------------

local Observer = torch.class("dp.Observer")

------------------------------------------------------------------------
--[[ LearningRateSchedule ]]--
-- Epoch Observer
-- Can be called from Propagator or Experiment
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

function LearningRateSchedule:onDoneEpoch(subject)
   local experiment
   if subject.isPropagator then
      experiment = subject:experiment()
   elseif subject.isExperiment then
      experiment = subject
   else
      error"Observer error: unsupported subject."
   end
   local epoch = experiment:epoch()
   local learning_rate = self._schedule[epoch]
   if learning_rate then
      experiment:optimizer():setLearningRate(learning_rate)
   end
end


------------------------------------------------------------------------
--[[ Experiment ]]--
-- Acts as a kind of Facade (Design Pattern) which inner objects can
-- use to access inner objects, i.e. objects used within the experiment.
------------------------------------------------------------------------

local Experiment = torch.class("dp.Experiment")

Experiment.isExperiment = true
Experiment.isMediator = true

function Experiment:__init(...)
   local args, datasource, optimizer, validator, tester, logger, 
      observers, random_seed, epoch, overwrite
      = xlua.unpack(
         {... or {}},
         'Experiment', nil,
         {arg='name', type='string'},
         {arg='datasource', type='dp.DataSource'},
         {arg='optimizer', type='dp.Optimizer'},
         {arg='validator', type='dp.Evaluator'},
         {arg='tester', type='dp.Evaluator'},
         {arg='logger', type='dp.Logger'},
         {arg='observers', type='dp.Observer'},
         {arg='random_seed', type='number', default=7},
         {arg='epoch', type='number', default=0,
          help='Epoch at which to start the experiment.'},
         {arg='overwrite', type='boolean', default=false,
          help='Overwrite existing values. For example, if a ' ..
          'datasource is provided, and optimizer is already ' ..
          'initialized with a dataset, and overwrite is true, ' ..
          'then optimizer would be setup with datasource:trainSet()'}
   )
   self._is_done_experiment = false
   self:setEpoch(epoch)
   self:setDatasource(datasource)
   self:setObservers(observers)
   if optimizer and datasource then
      optimizer:setup{experiment=self, 
                      dataset=datasource:trainSet(),
                      logger=optim.Logger(paths.concat(opt.save, 'train.log')),
                      overwrite=overwrite}
   end
   self:setOptimizer(optimizer)
   if validator and datasource then
      validator:setup{experiment=self, 
                      dataset=datasource:validSet(),
                      logger=optim.Logger(paths.concat(opt.save, 'valid.log')),
                      overwrite=overwrite}
   end
   self:setValidator(validator)
   if tester and datasource then
      tester:setup{experiment=self, 
                   dataset=datasource:testSet(),
                   logger=optim.Logger(paths.concat(opt.save, 'test.log')),
                   overwrite=overwrite}
   end
   self:setTester(tester)
end

function Experiment:run()
   self._validator:doEpoch()
   repeat
      self._epoch = self._epoch + 1
      self._optimizer:doEpoch()
      self._validator:doEpoch()
      self:doneEpoch()
   until self:isDoneExperiment()
   --test at the end
   self._tester:doEpoch()
end

--an observer should call this when the experiment is complete
function Experiment:doneExperiment()
   self._is_done_experiment = true
end

function Experiment:isDoneExperiment()
   return self._is_done_experiment
end

function Experiment:optimizer()
   return self._optimizer
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


function Experiment:doneEpoch()
   for i = 1,#self._epoch_observers do
      self._epoch_observers[i]:onDoneEpoch(self)
   end
end

function Experiment:setObservers(observers)
   self._observers = observers
   self._epoch_observers 
      = _.where(self._observers, {isEpochObserver=true})
end

function Experiment:observers()
   return self._observers
end

function Experiment:setEpoch(epoch)
   self._epoch = epoch
end

function Experiment:epoch()
   return self._epoch
end

function Experiment:setDatasource(datasource)
   self._datasource = datasource
end

function Experiment:datasource()
   return self._datasource
end

