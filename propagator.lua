require 'xlua'
require 'sys'
require 'torch'
require 'optim'

--[[ TODO ]]--
-- Logger (remove current logging)
-- Specify which values are default setup by Experiment in doc
-- Only one Observer per object
-- Mediator is referenced by all objects, including observers (setup)
-- Observers own concrete channels
-- Batch object. is it really necessary?
-- Propagator Strategies (Costs)

------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a model in order to evaluate its criteria, train the model,
-- etc.
------------------------------------------------------------------------

local Propagator = torch.class("dp.Propagator")

function Propagator:__init(...)   
   local args, model, sampler, logger, criterion,
      observer, mem_type, plot, progress
      = xlua.unpack(
      {... or {}},
      'Propagator', nil,
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='sampler', type='dp.Sampler', default=dp.Sampler(),
       help='used to iterate through the train set'},
      {arg='logger', type='dp.Logger',
       help='an object implementing the Logger interface'},
      {arg='criterion', type='nn.Criterion', req=true,
       help='a neural network criterion to evaluate or minimize'},
      {arg='observer', type='dp.Observer', 
       help='observers that are informed when an event occurs.'},
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='plot', type='boolean', default=false,
       help='live plot of confusion matrix'},
      {arg='progress', type'boolean', default=true, 
       help='display progress bar'}
   )
   sampler:setup{batch_size=batch_size, overwrite=false}
   self:setSampler(sampler)
   self:setModel(model)
   self:setMemType(mem_type)
   self:setCriterion(criterion)
   self:setObserver(observer)
   self:setPlot(plot)
   self:setProgress(progress)
   self._observer = observer
end

-- returns a log for the current epoch, in the format of a table
-- or we could create an EpochLog class to help with this.
-- But a table is more flexible. The advantage over pylearn2 is that 
-- the log of an epoch is structured, as opposed to just a list of 
-- channel names and values. Furthermore, values can be anything 
-- serializable.
function Propagator:report()
   local report = {
      name = self:name()
      sampler = self:sampler:report(),
      extra = {},
   }
end

function Propagator:setSampler(sampler)
   if self._sampler then
      -- initialize sampler with old sampler values without overwrite
      sampler:setup{batch_size=self:batchSize(), overwrite=false}
   end
   self._sampler = sampler
end

function Propagator:sampler()
   return self._sampler
end

function Propagate:observer()
   return self._observer
end

function Propagate:id()
   return self._id
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


function Propagator:setup(...)
   local args, id, model, logger, mem_type, overwrite
      = xlua.unpack(
      'Propagator:setup', nil,
      {arg='id', type='dp.ObjectID'},
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='mediator', type='dp.Mediator', req=true},
      {arg='logger', type='dp.Logger',
       help='an object implementing the Logger interface'},
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='overwrite', type='boolean', default=false,
       help='Overwrite existing values. For example, if a ' ..
       'dataset is provided, and sampler is already ' ..
       'initialized with a dataset, and overwrite is true, ' ..
       'then sampler would be setup with dataset'}
   )
   assert(id.isObjectID)
   self._id = id
   assert(mediator.isMediator)
   self._mediator = mediator
   self:setModel(model)
   self:setMemType(mem_type, overwrite)
   self:setLogger(logger, overwrite)
   
end

function Propagator:propagateEpoch()
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
end      

      
function Propagator:propagateBatch(batch)
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
   self._mediator:publish("doneBatch", report)
end

------------------------------------------------------------------------
--[[ PropagatorDecorator ]]--
-- Maintains a reference to a Propagator object and defines an interface
--- that conforms to Propagator's interface
------------------------------------------------------------------------

local PropagatorDecorator = torch.class("PropagatorDecorator", "Propagator")

function PropagatorDecorator:propagateBatch(batch, mediator)
   self._component:propagateBatch(batch, mediator)
end

function PropagatorDecorator:propagateEpoch(dataset, mediator)
   self._component:propagateEpoch(dataset, mediator)
end



