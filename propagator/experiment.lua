------------------------------------------------------------------------
--[[ Experiment ]]--
-- Acts as a kind of Facade (Design Pattern) which inner objects can
-- use to access inner objects, i.e. objects used within the experiment.
-- An experiment propagates DataSets through Models. The specifics 
-- such propagations are handled by Propagators. The propagation of 
-- a DataSet is called an epoch. At the end of each epoch, a monitoring
-- step is performed where.

-- We keep datasource/datasets, and mediator in outer scope to allow for 
-- experiment serialization. Serialization of an object requires 
-- that it be an instance of a registered torch.class. Sadly, objects 
-- which have references to functions (excluding those in their 
-- metatables) cannot currently be serialized. This also implies that 
-- the mediator cannot currently be serialized since it holds references
-- to callback functions. The latter could be swapped for objects and 
-- string function names to allow for mediator serialization. The data*
-- are kept out of scope in order to reduce the size of serializations.
-- Data* should therefore be reportless, i.e. restorable from 
-- its constructor.

-- The experiment keeps a log of the report of the experiment after 
-- every epoch. This is done by calling the report() method of 
-- every contained object, except Observers. The report is a read-only 
-- table that is passed to Observers along with the Mediator through 
-- Channels for which they are Subscribers. The report is also passed 
-- to sub-objects during propagation in case they need to act upon 
-- data found in other branches of the experiment tree.
------------------------------------------------------------------------

local Experiment = torch.class("dp.Experiment")
Experiment.isExperiment = true

function Experiment:__init(...)
   local args, id, id_gen, description, model, optimizer, validator, 
         tester, observer, random_seed, epoch, mediator, overwrite,
         max_epoch
      = xlua.unpack(
      {... or {}},
      'Experiment', nil,
      {arg='id', type='dp.ObjectID'},
      {arg='id_gen', type='dp.EIDGenerator'},
      {arg='description', type='string'},
      {arg='model', type='dp.Model', req=true},
      {arg='optimizer', type='dp.Optimizer'},
      {arg='validator', type='dp.Evaluator'},
      {arg='tester', type='dp.Evaluator'},
      {arg='observer', type='dp.Observer'},
      {arg='random_seed', type='number', default=7},
      {arg='epoch', type='number', default=0,
       help='Epoch at which to start the experiment.'},
      {arg='mediator', type='dp.Mediator', 
       help='defaults to dp.Mediator()'},
      {arg='overwrite', type='boolean', default=false,
       help='Overwrite existing values. For example, if a ' ..
       'datasource is provided, and optimizer is already ' ..
       'initialized with a dataset, and overwrite is true, ' ..
       'then optimizer would be setup with datasource:trainSet()'},
      {arg='max_epoch', type='number', default=1000, 
       help='Maximum number of epochs allocated to the experiment'}
   )
   self:setRandomSeed(random_seed)
   self._is_done_experiment = false
   assert(id or id_gen)
   self._id = id or id_gen:nextID()
   assert(self._id.isObjectID)
   self._model = model
   self._epoch = epoch
   self:setObserver(observer)
   self._optimizer = optimizer
   self._validator = validator
   self._tester = tester
   self._mediator = mediator or dp.Mediator()
   self:setMaxEpoch(max_epoch)
end

function Experiment:setup(datasource)
   --publishing to this channel will terminate the experimental loop
   self._mediator:subscribe("doneExperiment", self, "doneExperiment")
   -- Even though the model is used by different propagators, the 
   -- model uses the experiment's namespace:
   self._model:setup{mediator=self._mediator, 
                     id=self:id():create('model')}
   if self._optimizer then
      self._optimizer:setup{mediator=self._mediator, model=self._model,
                            id=self:id():create('optimizer'),
                            dataset=datasource:trainSet()}
   end
   if self._validator then
      self._validator:setup{mediator=self._mediator, model=self._model,
                            id=self:id():create('validator'),
                            dataset=datasource:validSet()}
   end
   if self._tester then
      self._tester:setup{mediator=self._mediator, model=self._model,
                            id=self:id():create('tester'),
                            dataset=datasource:testSet()}
   end
   if self._observer then
      self._observer:setup{mediator=self._mediator, subject=self}
   end
   self._setup = true
end

--TODO : make this support explicit dataset specification (xlua.unpack)
--loops through the propagators until a doneExperiment is received or
--experiment reaches max_epochs
function Experiment:run(datasource)
   if not self._setup then
      self:setup(datasource)
   end
   -- use mediator to publishes 'doneEpoch' for observers and allows
   -- observers to communicate with each other. Should only be used 
   -- during the monitoring phase
   
   -- gets a read-only copy of all experiment parameters
   local report = self:report()
   local train_set = datasource:trainSet()
   local valid_set = datasource:validSet()
   local test_set = datasource:testSet()
   repeat
      self._epoch = self._epoch + 1
      self._optimizer:propagateEpoch(train_set, report)
      self._validator:propagateEpoch(valid_set, report)
      self._tester:propagateEpoch(test_set, report)
      report = self:report()
      self._mediator:publish("doneEpoch", report)
   until (self:isDoneExperiment() or self._epoch >= self._max_epoch)
   self._mediator:publish("finalizeExperiment")
end

--an observer should call this when the experiment is complete
function Experiment:doneExperiment()
   self._is_done_experiment = true
end

function Experiment:isDoneExperiment()
   return self._is_done_experiment
end

function Experiment:id()
   return self._id
end

function Experiment:name()
   return self._id:name()
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

function Experiment:randomSeed()
   return self._random_seed
end

function Experiment:report()
   local report = {
      optimizer = self:optimizer():report(),
      validator = self:validator():report(),
      tester = self:tester():report(),
      epoch = self:epoch(),
      random_seed = self:randomSeed(),
      model = self._model:report(),
      id = self._id:toString(),
      description = description
   }
   return report
end

function Experiment:observers()
   return self._observers
end

function Experiment:epoch()
   return self._epoch
end

function Experiment:setMaxEpoch(max_epoch)
   self._max_epoch = max_epoch
end

function Experiment:setObserver(observer)
   if not torch.typename(observer) and type(observer) == 'table' then
      --if list, make composite observer
      observer = dp.CompositeObserver(observer)
   end
   self._observer = observer
end

function Experiment:setRandomSeed(random_seed)
   torch.manualSeed(random_seed)
   math.randomseed(random_seed)
   self._random_seed = random_seed
end
