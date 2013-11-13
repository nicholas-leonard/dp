require 'xlua'
require 'sys'
require 'torch'
require 'optim'

--[[ TODO ]]--
-- Logger (remove current logging)
-- Specify which values are default setup by Experiment

--[[ Discussion ]]--
--[[ 
The EarlyStopper needs the validation error. More specifically, 
it needs to early stop on some value that should be minimized.

It is initialized with a function that is given the experiment as a 
parameter in the hope that this will prove flexible enough that users 
can early stop on any metric.

We could create an object similar to pylearn2 costs, which takes 
outputs (predictions) and targets as input and compares them to 
generate a value. The ConfusionMatrix kind of does this through 
batchAdd(outputs, targets). 

So these Cost objects could be BatchObservers if the batch outputs
and targets would be made available to them via the experiment.

The experiment acts as a kind of host for all states of the experiment.
By passing experiment to observers, these can acess all data made 
available in the experiment's interface.

The main issue is that observers could easily be implemented that 
call validator:doEpoch(), optimizer:doEpoch(), etc. Therefore, 
breaking the normal flow of learning. But then again, lua is not about
protecting anything. And we want observers to be able to access and 
even call anything. Users can then build very powerful observers. 

Some Propagators or Experiments can become default initialized with 
a set of commonly used observers to help new beginners.

So what? Propagators, Experimentors, i.e. objects made accessible 
to the observers should provide interfaces for all the state 
information that they store. 

Okay, but what about the information that observers store within 
themselves, should we allow observers to access each other's states?
The problem with this is that the order observers are updated is not 
garanteed, such that some observers reading the state of another 
observer may be acting on data not yet up-to-date.

We could create an Epoch and Batch object that is passed to 
all EpochObservers and BatchObservers respectively? Maybe later...
]]--
------------------------------------------------------------------------
--[[ EarlyStopper ]]--
-- Epoch Observer.
-- Saves model with the lowest validation error (default). 
------------------------------------------------------------------------

local EarlyStopper = torch.class("dp.EarlyStopper")

EarlyStopper.isEpochObserver = true

function EarlyStopper:__init(...)
   local valid_error = function(experiment)
      return experiment:validator():error()
   end
   local args, start_epoch, error_func = xlua.unpack(
      'EarlyStopper', nil,
      {arg='start_epoch', type='number', default=5,
       help='when to start saving models.'},
      {arg='error_func', type='function', default=valid_error,
       help='Function called on experiment. Returns and error.'}
   )
   self._start_epoch = start_epoch
   self._error_func = error_func
end

function EarlyStopper:onDoneEpoch(subject)
   error"Debug this"
   local current_error = self._error_func(experiment)
   local epoch = experiment:epoch()
   if epoch >= start_epoch then
      if (not self._minima) or (current_error > self._minima) then
         self._minima = current_error
         --TODO save model
      end
   end
      
   local epoch = experiment:epoch()
   local learning_rate = self._schedule[epoch]
   if learning_rate then
      experiment:optimizer():setLearningRate(learning_rate)
   end
   -- save/log current net
   local filename = paths.concat(opt.save, self._name .. '_model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

end

------------------------------------------------------------------------
--[[ Model ]]--
-- Encapsulates a nn.Module such that it can be used for both 
-- optimzation and evaluation.
------------------------------------------------------------------------

------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a model in order to evaluate its criteria, train the model,
-- etc.
------------------------------------------------------------------------

local Propagator = torch.class("dp.Propagator")

function Propagator:__init(...)   
   local args, dataset, model, batch_size, mem_type, logger, criterion,
      observers, mem_type, plot, progress
      = xlua.unpack(
      {... or {}},
      'Propagator', nil,
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='sampler', type='dp.Sampler', default=dp.Sampler(),
       help='used to iterate through the train set'},
      {arg='dataset', type='dp.Dataset',
       help='used to setup sampler if not done already'},
      {arg='logger', type='dp.Logger',
       help='an object implementing the Logger interface'},
      {arg='criterion', type='nn.Criterion', req=true,
       help='a neural network criterion to evaluate or minimize'},
      {arg='observers', type='dp.Observer', 
       help='observers that are informed when an event occurs.'},
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='plot', type='boolean', default=false,
       help='live plot of confusion matrix'},
      {arg='progress', type'boolean', default=true, 
       help='display progress bar'},
      {arg='name', type='string',
       help='identifies the propagator. ' ..
       'Common values: "train", "valid", "test"'}
   )
   sampler:setup{dataset=dataset, 
                 batch_size=batch_size, 
                 overwrite=false}
   self:setSampler(sampler)
   self:setModel(model)
   self:setMemType(mem_type)
   self:setCriterion(criterion)
   self:setObservers(observers)
   self:setLogger(logger)
   self:setPlot(plot)
   self:setProgress(progress)
   self:setName(name)
end

function Propagator:setSampler(sampler)
   if self._sampler then
      -- initialize sampler with old sampler values without overwrite
      sampler:setup{dataset=self._sampler:dataset(), 
                    batch_size=self._sampler:batchSize(),
                    overwrite=false}
   end
   self._sampler = sampler
end

function Propagator:sampler()
   return self._sampler
end

function Propagator:setDataset(dataset, overwrite)
   self._sampler:setDataset(dataset, overwrite)
end

function Propagator:dataset()
   return self._sampler:dataset()
end

function Propagator:setModel(model)
   self._model = model
end

function Propagator:model()
   return self._model
end

function Propagator:setBatchSize(batch_size, overwrite)
   self._sampler:setBatchSize(batch_size, overwrite)
end

function Propagator:batchSize()
   return self._sampler:batchSize()
end

function Propagator:setExperiment(experiment)
   self._sampler:setExperiment(experiment)
   self._experiment = experiment
end

function Propagator:experiment()
   return self._experiment
end

function Propagator:epoch()
   return self:experiment():epoch()
end

function Propagator:setLogger(logger, overwrite)
   if (overwrite or not self._logger) and logger then
      self._logger = logger
   end
end

function Propagator:logger()
   return self._logger
end

function Propagator:setCriterion(criterion)
   self._criterion = criterion
end

function Propagator:criterion()
   return self._criterion
end

function Propagator:setMemType(mem_type, overwrite)
   if (overwrite or not self._mem_type) and mem_type then
      self._mem_type = mem_type
   end
end

function Propagator:memType()
   return self._mem_type
end

--TODO : need a way to append/extend observers
function Propagator:setObservers(observers, overwrite)
   if (overwrite or not self._observers) and observers then
      self._observers = observers
      self._epoch_observers 
         = _.where(self._observers, {isEpochObserver=true})
      self._batch_observers
         = _.where(self._observers, {isBatchObserver=true})
   end
end

function Propagator:observers()
   return self._observers
end

function Propagator:setName(name, overwrite)
   if (overwrite or not self._name) and name then
      self._name = name
   end
end

function Propagator:name()
   return self._name
end

--should be called when batch is complete
function Propagator:doneBatch()
   for i = 1,#self._batch_observers do
      self._batch_observers[i]:onDoneBatch(self)
   end
end

--should be called when epoch is complete
function Propagator:doneEpoch()
   for i = 1,#self._epoch_observers do
      self._epoch_observers[i]:onDoneEpoch(self)
   end
end

function Propagator:setup(...)
   local args, experiment, model, dataset, logger, mem_type, overwrite
      = xlua.unpack(
      'Propagator:setup', nil,
      {arg='experiment', type='dp.Experiment', req=true,
       help='Acts as a Mediator (design pattern). ' ..
       'Provides access to the experiment.'},
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='dataset', type='dp.Dataset',
       help='used to setup sampler if not done already'},
      {arg='logger', type='dp.Logger',
       help='an object implementing the Logger interface'},
      {arg='observers', type='dp.Observer', 
       help='observers that are informed when an event occurs.'},
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='overwrite', type='boolean', default=false,
       help='Overwrite existing values. For example, if a ' ..
       'dataset is provided, and sampler is already ' ..
       'initialized with a dataset, and overwrite is true, ' ..
       'then sampler would be setup with dataset'},
      {arg='name', type='string',
       help='identifies the propagator. ' ..
       'Common values: "train", "valid", "test"'}
   )
   self._sampler:setup{experiment=experiment, 
                       dataset=dataset, 
                       overwrite=overwrite}
   self:setExperiment(experiment)
   self:setModel(model)
   self:setMemType(mem_type, overwrite)
   self:setObservers(observers, overwrite)
   self:setLogger(logger, overwrite)
   self:setName(name, overwrite)
end

function Propagator:doEpoch()
   -- TODO refactor into default datasource:observers()?
   local confusion = optim.ConfusionMatrix2(self._experiment:datasource():classes())
   
   -- local vars
   local time = sys.clock()
   -- do one epoch
   print('==> doing epoch on training data:')
   print('==> online epoch # ' .. self:epoch() .. 
         ' [batchSize = ' .. self:batchSize() .. ']')
         
   for batch in self:sampler():sampleEpoch() do
      if self._progress then
         -- disp progress
         xlua.progress(start, self:dataset():size())
      end
      self:doBatch(batch)
   end
   -- time taken
   time = sys.clock() - time
   time = time / self:dataset():size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   
   local title = '% mean class accuracy ( ' .. self._name .. ' )'
   -- update logger/plot
   self._logger:add{[title] = confusion.totalValid * 100}
   if self._plot then
      self._logger:style{[title] = '-'}
      self._logger:plot()
   end

   -- next epoch
   confusion:zero()
   self:doneEpoch()
end      

      
function Propagator:doBatch(batch)
   local inputs = batch:inputs()
   local targets = batch:targets()

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
        -- get new parameters
        if x ~= parameters then
           parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        --[[feedforward]]--
        -- evaluate function for complete mini batch
        local outputs = model:forward(inputs)
        -- average loss (a scalar)
        local f = criterion:forward(outputs, targets)
        
        --[[backpropagate]]--
        -- estimate df/do (o is for outputs), a tensor
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
         
        --[[measure error]]--
        -- update confusion
        confusion:batchAdd(outputs, targets)
                       
        -- return f and df/dX
        return f,gradParameters
     end

   optim.sgd(feval, parameters, optimState)
   self:doneBatch()
end

------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------

local Optimizer = torch.class("dp.Optimizer", "dp.Propagator")

function Optimizer:__init(...)
   local args, sampler, learning_rate, weight_decay, momentum
      = xlua.unpack(
      {... or {}},
      'Optimizer', 
      'Optimizes a model on a training dataset',
      {arg='sampler', type='dp.Sampler', default=dp.ShuffleSampler(),
       help='used to iterate through the train set'},
      {arg='learning_rate', type='number', req=true,
       help='learning rate at start of learning'},
      {arg='weight_decay', type='number', default=0,
       help='weight decay coefficient'},
      {arg='momentum', type='number', default=0,
       help='momentum of the parameter gradients'}
   )
   self:setLearningRate(learning_rate)
   self:setWeightDecay(weight_decay)
   self:setMomentum(momentum)
   Propagator.__init(self, ...)
   self:setSampler(sampler)
end

function Optimizer:doBatch(batch)
   local inputs = batch:inputs()
   local targets = batch:targets()

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
        -- get new parameters
        if x ~= parameters then
           parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        --[[feedforward]]--
        -- evaluate function for complete mini batch
        local outputs = self._model:forward(inputs)
        -- average loss (a scalar)
        local f = self._criterion:forward(outputs, targets)
        
        --[[backpropagate]]--
        -- estimate df/do (o is for outputs), a tensor
        local df_do = self._criterion:backward(outputs, targets)
        self._model:backward(inputs, df_do)
         
        --[[measure error]]--
        -- update confusion
        confusion:batchAdd(outputs, targets)
                       
        -- return f and df/dX
        return f, gradParameters
     end

   optim.sgd(feval, parameters, optimState)
   self:doneBatch()
end

------------------------------------------------------------------------
--[[ Evaluator ]]--
-- Tests (evaluates) a model using a sampling distribution.
-- For evaluating the generalization of the model, seperate the 
-- training data from the test data. The Evaluator can also be used 
-- for early-stoping.
------------------------------------------------------------------------


local Evaluator = torch.class("dp.Evaluator", "dp.Propagator")

function Evaluator:doBatch(batch)
   local inputs = batch:inputs()
   local targets = batch:targets()
   -- get new parameters
   if x ~= parameters then
     parameters:copy(x)
   end

   -- reset gradients
   gradParameters:zero()

   --[[feedforward]]--
   -- evaluate function for complete mini batch
   local outputs = self._model:forward(inputs)
   -- average loss (a scalar)
   local f = self._criterion:forward(outputs, targets)

   --[[measure error]]--
   -- update confusion
   confusion:batchAdd(outputs, targets)
   self:doneBatch()
end

