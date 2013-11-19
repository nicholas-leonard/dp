require 'torch'

require 'utils'

------------------------------------------------------------------------
--[[ Signal ]]--
-- A signal is passed along a Chain of responsibility (design pattern) 
-- as a request. It can be decorated (design pattern) by any Node in 
-- the chain. It can be composed of sub-signals. It has a log interface 
-- that can be updated by any Node. 
------------------------------------------------------------------------
local Signal = torch.class("dp.Signal")

function Signal:__init(...)
   self._task = 'Setup'
   self._data = {}
end

function Signal:task()
   return self._task
end

function Signal:extend(metatable)
   --TODO find way to hide non-serializable variables (datasource)
   --using the metatable. 
end
------------------------------------------------------------------------
--[[ Node ]]--
-- A node handles Signals by updating its state and passes it along 
-- to the calling Chain. A node is a component of a Chain, which is 
-- a composite (design pattern) of Nodes. Nodes cannot see each other. 
-- They communicate through the Signal.
------------------------------------------------------------------------
local Node = torch.class("dp.Node")

function Node:__init(name)
   self._name = "NoName"
   self._isSetup = false
end

function Node:name()
   return self._name
end

-- setup the object.
-- by default a chain is setup following construction by calling
-- propagating a signal with task="Setup"
function Node:handleSetup(signal)
   if (not self._isSetup) and self.setup then
      return self:setup(signal)
   end
   return signal, true
end

function Node:handleTask(signal)
   -- call a function by the name of task
   local handler = self['handle' .. signal:task()]
   if handler then
      return handler(self, signal)
   end
   return signal, false
end

function Node:handleSignal(signal)
   local signal, handled = self:handleTask(signal)
   if not handled then
      signal, handled = self:handleAny(signal)
      print"Warning: Unhandled signal by Node " .. self:name() ..
         " for task " .. signal:task() .. ". Maybe signal was " .. 
         "handled but didn't notify the caller?")
   end
   return signal, handled
end

function Node:handleAny(signal)
   -- default is to enforce task handling
   error("Error: unhandled signal by Node " .. self:name() ..
      " for task " .. signal:task() .. ". Consider implementing " .. 
      "a handleAny method for handling unknown tasks.")
end
   


------------------------------------------------------------------------
--[[ Chain ]]--
-- A chain is also a Node, but it is composed of a list of Nodes to be 
-- iterated (composite design pattern). A node can be referenced many 
-- time in a chain.
------------------------------------------------------------------------
local Chain = torch.class("dp.Chain", "dp.Node")

function Chain:__init(name, nodes)
   self._nodes = nodes
   Node:__init(name)
end

function Chain:nodes()
   return self._nodes
end

--[[function Chain:handleSetup(signal)
   signal
   return signal, false
end]]--

function Chain:handleAny(signal)
   local handled
   local saved_signal = signal
   -- iterate through the chain of nodes
   for i, node in ipairs(self:nodes()) do
      signal, handled = node:handleSignal(signal)
   end
end



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

local ExperimentState = {
   _optimizer={}, 
   _validator={}, 
   _tester={},
   _model={}
}

function ExperimentState:optimizer()
   return self._optimizer
end

function ExperimentState:validator()
   return self._validator
end

function ExperimentState:tester()
   return self._tester
end

function ExperimentState:model()
   return self._model
end

function ExperimentState:datasource()
   return self._datasource
end

local Experiment = torch.class("dp.Experiment", "dp.Chain")

Experiment.isExperiment = true

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
         {arg='random_seed', type='number', default=7},
         {arg='epoch', type='number', default=0,
          help='Epoch at which to start the experiment.'},
         {arg='overwrite', type='boolean', default=false,
          help='Overwrite existing values. For example, if a ' ..
          'datasource is provided, and optimizer is already ' ..
          'initialized with a dataset, and overwrite is true, ' ..
          'then optimizer would be setup with datasource:trainSet()'}
   )
   self:setEpoch(epoch)
   self:setDatasource(datasource)
   self:setObservers(observers)
   local nodes = {}
   if optimizer and datasource then
      optimizer:setup{experiment=self, name='optimizer'
                      dataset=datasource:trainSet(),
                      logger=optim.Logger(paths.concat(opt.save, 'train.log')),
                      overwrite=overwrite}
   end
   if optimizer then
      table.insert(nodes, optimizer)
   end
   if validator and datasource then
      validator:setup{experiment=self, name='validator',
                      dataset=datasource:validSet(),
                      logger=optim.Logger(paths.concat(opt.save, 'valid.log')),
                      overwrite=overwrite}
   end
   if validator then
      table.insert(nodes, validator)
   end
   if tester and datasource then
      tester:setup{experiment=self, name='tester',
                   dataset=datasource:testSet(),
                   logger=optim.Logger(paths.concat(opt.save, 'test.log')),
                   overwrite=overwrite}
   end
   if tester then
      table.insert(nodes, tester)
   end
   Chain.__init("Experiment", {optimizer, validator, tester})
end

function Experiment:setup(signal)
   signal:extend(ExperimentState)
   -- makes the _datasource unserializable (coule be second argument)
   signal:extend({_datasource=self._datasource})
end

function Experiment:run()
   local signal = {}
   signal = self:handleSetup(signal)
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


function Experiment:setDatasource(datasource)
   self._datasource = datasource
end

function Experiment:datasource()
   return self._datasource
end

--The problem is that an object must know its location in the tree
--SetNamespace ExtendNamespace (special nodes can change theses like validators, or just dedicated nodes)
--Propagators just change the state (which is a position in the tree)
-- but still have access to the remainder of the tree (current, parent, root, child)
-- what about objects like models that can be in many places?
-- they are to be stored in the highest level where it can be unique
-- all nodes are like Commands that have access to there state and the
-- state of the tree (root). The are called with execute() in order, 
-- which allows them to modify the state before it is executed by the next one.
-- Nevertheless, each Node is oblivious to its predecessors in the chain
-- it knows only of the current task or mode, which will determine its behavior,
-- and of the state of the system, which it will use to perform its task.
--An issue is how to limit the granularity of this task. For example, 
-- a propagator's doBatch and doEpoch could be divided into very small 
-- Nodes. Which would offer great flexibility in all, but this would 
-- eventually become analogous to a language, requiring an interpreter.
--Building models would be very finegrained and ultimately feel like procedural or
-- functional programming. 

--So do we want OOP or a pipeline?
--If we refactor our propagators and such sufficiently, and allow for 
--a means of decorating propagators, we will have the same result.
--The problem is that we cannot serialize what we decorate. As soon as 
-- we change the metatable, the torch.factory cannot rebuild it during
-- unserialization. For that we would have to build our own decorator 
-- factory. The typename is serialized and used to setup the metatable
-- during serialization. We could modify the serializer to serialize 
-- all decorators, which would have to be made available globally.
-- Decorators would not be torch classes. They would have a special type.
-- But for now the easiest thing to do is to hard code what we need.
-- Propagators can have a chain of responsibility to pass on various 
-- requests. For example, they can pass on a cost request holding 
-- targets and predictions.

--Modules can still be encapsulated by Layers that provide 
--additional facilities like logging, momentum, max col norm, etc. 
--These could also be decorated (not yet).
--The model would be its own object of layers, and itself a layer.
--It would be serializable.

--We still have the problem of learning rate modifications. 
--We build a StateOfTheEpoch and pass it along the tree and handlers 
-- to see if anyone needs it or needs to modify it.
--So we still need a way of implementing a global state that we can 
-- pass along the tree. Every component, because the tree is just a 
-- composite pattern, gets called (state = onEpoch(state)) which they forward to 
-- there components in a logical order. State data is identifies as log / or not
-- All extensions have name and decorate this function? That would be 
-- ideal.

-- The log is simply a matter of initializing objects with a 
-- namespace. This is used to insert their entry into the log.
-- There is only one log each epoch, none for batches.
-- Each log is a bunch of leafs identifies by a namespace and name.
-- A namespace is a chain of names. 
-- The namespace must be standardized to facilitate analysis.
-- A log_entry is not limited to any one namespace? So for example, 
-- I want to log layers as they are structured in the model (in which 
-- position), but also based on their Module class. Nah. The class of 
-- the layer should be logged such that we can later browse trees for 
-- class level statistics. So we should store the typename. As well 
-- as the parent typenames.

--Once the state is passed along the subtrees, it is used by the 
-- components to forward batches and so on. This state contains 
-- is easily namespaced, just like the log. Actually they are the same.
-- Except that some variables are specified as not-for-logging.
-- Objects may be initialized with a log-filter. A default is provided.
-- Its basically a namespace tree whose leafs's childrent in the state 
-- tree are not logged. Or the filter can say which ones to log.
--So the state and log are the same. A simple table that can be traversed.
-- and serialized. It should contain all data. Some of it should be 
-- marked as unserializable.

--Objects should allow for being setup with such a state. But the 
-- datasource shouldn't be serialized. And the state shouldn't contain 
-- parameters, these are in the model. It should be divided into 
-- constructor, epoch, batch, destructor namespaces. The batch should
-- not be accessible between propagators, it is only used as a convenience.
-- The constructor contains the design (see Builder). And the epoch 
-- contains the data required for logging each epoch, and for each 
-- component. It keeps track of changes in the hyper-parameters, and 
-- also keeps a log of observations (error, col norm, etc). The 
-- destructor is for anything that occurs after all epochs are complete.
-- We would also require the model be clonable for early-stopping. 
-- We would like to keep it memory for final testing. Another possibility,
-- is to perform testing only when the validator finds a new minima.
-- state.epoch.validator.minimum_error=4353
-- state.epoch.validator.current_error=2323 --new minima! do something...

