------------------------------------------------------------------------
--[[ Experiment ]]--
-- An experiment propagates DataSets through Models. The specifics 
-- such propagations are handled by Propagators. The propagation of 
-- a DataSet is called an epoch. At the end of each epoch, a monitoring
-- step is performed where a report is generated for all observers.

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

function Experiment:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, id, model, target_module, optimizer, validator, tester, 
      observer, random_seed, epoch, mediator, max_epoch, description
      = xlua.unpack(
      {config},
      'Experiment', 
      'An Experiment to be performed using a Model, Propagators '..
      'and a DataSource',
      {arg='id', type='dp.ObjectID',
       help='uniquely identifies the experiment. '..
       'Defaults to using dp.uniqueID() to initialize a dp.ObjectID'},
      {arg='model', type='nn.Module', req=true,
       help='Module instance shared by all Propagators.'},
      {arg='target_module', type='nn.Module', 
       help='Optional module through which targets can be forwarded'},
      {arg='optimizer', type='dp.Optimizer',
       help='Optimizer instance used for propagating the train set'},
      {arg='validator', type='dp.Evaluator', 
       help='Evaluator instance used for propagating the valid set'},
      {arg='tester', type='dp.Evaluator',
       help='Evaluator instance used for propagating the test set'},
      {arg='observer', type='dp.Observer', 
       help='Observer instance used for extending the Experiment'},
      {arg='random_seed', type='number', default=os.time(),
       help='number used to initialize the random number generator'},
      {arg='epoch', type='number', default=0,
       help='Epoch at which to start the experiment.'},
      {arg='mediator', type='dp.Mediator', 
       help='used for inter-object communication. defaults to dp.Mediator()'},
      {arg='max_epoch', type='number', default=1000, 
       help='Maximum number of epochs allocated to the experiment'},
      {arg='description', type='string', default='',
       help='A short description of the experiment'}
   )
   self:randomSeed(random_seed)
   self:id(id or dp.ObjectID(dp.uniqueID()))
   self:model(model)
   self:epoch(epoch)
   self:observer(observer)
   self:optimizer(optimizer)
   self:validator(validator)
   self:tester(tester)
   self:mediator(mediator or dp.Mediator())
   self:maxEpoch(max_epoch)
   self._target_module = target_module
   self._is_done_experiment = false
   self._description = description
end

function Experiment:setup()
   --publishing to this channel will terminate the experimental loop
   self._mediator:subscribe("doneExperiment", self, "doneExperiment")
   if self._optimizer then
      self._optimizer:setup{
         mediator=self._mediator, id=self:id():create('optimizer'),
         model=self._model, target_module=self._target_module
      }
   end
   if self._validator then
      self._validator:setup{
         mediator=self._mediator, id=self:id():create('validator'),
         model=self._model, target_module=self._target_module
      }
   end
   if self._tester then
      self._tester:setup{
         mediator=self._mediator, id=self:id():create('tester'),
         model=self._model, target_module=self._target_module
      }
   end
   if self._observer then
      self._observer:setup{mediator=self._mediator, subject=self}
   end
   self._setup = true
end

--loops through the propagators until a doneExperiment is received or
--experiment reaches max_epochs
function Experiment:run(datasource, once)
   if not self._setup then
      self:setup()
   end   
   local report = self:report()
   local train_set = datasource:trainSet()
   local valid_set = datasource:validSet()
   local test_set = datasource:testSet()
   local atleastonce = false
   repeat
      self._epoch = self._epoch + 1
      if self._optimizer and train_set then
         self._optimizer:propagateEpoch(train_set, report)
      end
      if self._validator and valid_set then
         self._validator:propagateEpoch(valid_set, report)
      end
      if self._tester and test_set then
         self._tester:propagateEpoch(test_set, report)
      end
      report = self:report()
      self._mediator:publish("doneEpoch", report)
      atleastonce = true
   until (self:isDoneExperiment() or self._epoch >= self._max_epoch or (once and atleastonce))
   self._mediator:publish("finalizeExperiment")
end

--an observer should call this when the experiment is complete
function Experiment:doneExperiment()
   self._is_done_experiment = true
end

function Experiment:isDoneExperiment()
   return self._is_done_experiment
end

function Experiment:report()
   local report = {
      optimizer = self:optimizer() and self:optimizer():report(),
      validator = self:validator() and self:validator():report(),
      tester = self:tester() and self:tester():report(),
      epoch = self:epoch(),
      random_seed = self:randomSeed(),
      id = self._id:toString(),
      description = self._description
   }
   return report
end

function Experiment:id(id)
   if id then
      assert(torch.isTypeOf(id, 'dp.ObjectID'), "Expecting dp.ObjectID instance")
      self._id = id
      return
   end
   return self._id
end

function Experiment:name()
   return self._id:name()
end

function Experiment:model(model)
   if model then
      assert(torch.isTypeOf(model, 'nn.Module'), "Expecting nn.Module instance")
      if not torch.isTypeOf(model, 'nn.Serial') then
         self._model = nn.Serial(model)
         self._model:mediumSerial(false)
      else
         self._model = model
      end
      return
   end
   return self._model
end

function Experiment:optimizer(optimizer)
   if optimizer then
      self._optimizer = optimizer
      return
   end
   return self._optimizer
end

function Experiment:validator(validator)
   if validator then
      self._validator = validator
      return
   end
   return self._validator
end

function Experiment:tester(tester)
   if tester then
      self._tester = tester
      return
   end
   return self._tester
end

function Experiment:mediator(mediator)
   if mediator then
      self._mediator = mediator
      return
   end
   return self._mediator
end

function Experiment:randomSeed(random_seed)
   if random_seed then
      torch.manualSeed(random_seed)
      math.randomseed(random_seed)
      self._random_seed = random_seed
      return 
   end
   return self._random_seed
end

function Experiment:epoch(epoch)
   if epoch then
      self._epoch = epoch
      return 
   end
   return self._epoch
end

function Experiment:maxEpoch(max_epoch)
   if max_epoch then
      self._max_epoch = max_epoch
      return
   end
   return self._max_epoch
end

function Experiment:observer(observer)
   if observer then
      if torch.type(observer) == 'table' then
         -- if list, make composite observer
         observer = dp.CompositeObserver(observer)
      end
      self._observer = observer
      return 
   end
   return self._observer
end

function Experiment:includeTarget(mode)
   if self._optimizer then
      self._optimizer:includeTarget(mode)
   end
   if self._validator then
      self._validator:includeTarget(mode)
   end
   if self._tester then
      self._tester:includeTarget(mode)
   end
end

function Experiment:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
   if self._optimizer then
      self._optimizer:verbose(self._verbose)
   end
   if self._validator then
      self._validator:verbose(self._verbose)
   end
   if self._tester then 
      self._tester:verbose(self._verbose)
   end
   if self._observer then
      self._observer:verbose(self._verbose)
   end
end

function Experiment:silent()
   self:verbose(false)
end

function Experiment:type(new_type)
   self._model:mediumSerial(false)
   if self._model then
      self._model:type(new_type)
   end
   if self._optimizer then
      self._optimizer:type(new_type)
   end
   if self._validator then
      self._validator:type(new_type)
   end
   if self._tester then
      self._tester:type(new_type)
   end
end

function Experiment:float()
   return self:type('torch.FloatTensor')
end

function Experiment:double()
   return self:type('torch.DoubleTensor')
end

function Experiment:cuda()
   return self:type('torch.CudaTensor')
end
