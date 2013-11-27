require 'xlua'
require 'sys'
require 'torch'
require 'optim'

--[[ TODO ]]--
-- Convolution (way later)
-- torch.clip

-- Specify which values are default setup by Experiment in doc
-- Make reports read-only (everywhere)
-- Observers and Feedback lists are automatically converted to composites
-- All objects are setup with random seed and mediator and data*
-- Input Model (minimum) Output, Feedback (bonus) or Feedback container (propagator...)
-- Propagator takes gstate
-- Model accepts observers?
-- Tests are performed every early stop?
-- 'Overwrite' arg for setup
-- Experiment is just a composite Propagator?

-- Optimizer requires visitor.
-- Visitors require report names to make them accessible to other observers, etc. like feedback
-- No need for visitor id. Ids are for serialization.
-- Do visitors gather statistics stored in model statistics?
-- In, out axes for model params
-- Prop, Model, Visitor namespace for prop.mvstate_matrix

------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a model in order to evaluate its criteria, train the model,
-- etc.
-- To make your own training algorithm, you can build your own 
-- propagator. If you can reduce it to reusable components, you could 
-- then refactor these into visitors, observers, etc.
------------------------------------------------------------------------

local Propagator = torch.class("dp.Propagator")
Propagator.isPropagator = true

function Propagator:__init(...)   
   local args, sampler, criterion, visitor, observer, feedback, 
         mem_type, progress
      = xlua.unpack(
      {... or {}},
      'Propagator', nil,
      {arg='sampler', type='dp.Sampler', default=dp.Sampler(),
       help='used to iterate through the train set'},
      {arg='criterion', type='nn.Criterion', req=true,
       help='a neural network criterion to evaluate or minimize'},
      {arg='visitor', type='dp.Visitor', req=true,
       help='visits models as chain links ' ..
       'during setup, forward, backward and update phases'},
      {arg='observer', type='dp.Observer', 
       help='observer that is informed when an event occurs.'},
      {arg='feedback', type='dp.Feedback',
       help='takes predictions, targets, model and visitor as input ' ..
       'and provides feedback through report(), setState, or mediator'},
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='progress', type'boolean', default=true, 
       help='display progress bar'}
   )
   self:setSampler(sampler)
   self:setMemType(mem_type)
   self:setCriterion(criterion)
   self:setObserver(observer)
   self:setProgress(progress)
   self._feedback = feedback
   self:setVisitor(visitor)
   self._progress = progress
   
   self:resetLoss()
end


function Propagator:setup(...)
   local args, id, model, logger, mem_type, overwrite
      = xlua.unpack(
      {... or {}},
      'Propagator:setup', nil,
      {arg='id', type='dp.ObjectID'},
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='mediator', type='dp.Mediator', req=true},
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
   if self._observer then
      self._observer:setup{mediator=mediator, subject=self}
   end
   if self._feedback then
      self._feedback:setup{mediator=mediator, propagator=self}
   end
   if self._visitor then
      self._visitor:setup{mediator=mediator, 
                          model=self.model, 
                          propagator=self}
   end
end

function Propagator:propagateEpoch(dataset, report)
   self._feedback:setup{dataset=dataset, report=report, 
                        mediator=self._mediator, model=self._model,
                        visitor=self._visitor}
   self:resetLoss()
   
   -- local vars
   local time = sys.clock()
   local batch
   
   print('==> doing epoch on ' .. self:id():name() .. ' data:')
   print('==> online epoch # ' .. report.epoch )
         
   for batch in self:sampler():sampleEpoch(dataset) do
      if self._progress then
         -- disp progress
         xlua.progress(batch:batchIter(), batch:epochSize())
      end
      self:propagateBatch(batch)
   end
   
   -- time taken
   self._epoch_duration = sys.clock() - time
   self._batch_duration = epoch_time / batch:epochSize()
   self._example_speed = batch:epochSize()/self._batch_durtion
   self._num_batches = batch:epochSize()/self:batchSize()
   self._batch_speed = (self._num_batches/self._batch_duration)
   print("\n==> batch duration = " .. 
         (self._batch_duration*1000) .. 'ms')
   print("==> epoch duration = " .. 
         (self._epoch_duration) .. 's')
   print("==> example speed = " .. 
         (self._example_speed) .. 'examples/second')
   print("==> batch speed = " .. 
         (self._batch_speed) .. 'batches/second')
end      
      
function Propagator:propagateBatch(batch)
   error"NotImplementedError"
end

-- returns a log for the current epoch, in the format of a table
-- or we could create an EpochLog class to help with this.
-- But a table is more flexible. The advantage over pylearn2 is that 
-- the log of an epoch is structured, as opposed to just a list of 
-- channel names and values. Furthermore, values can be anything 
-- serializable.
function Propagator:report()
   local report = {
      name = self:id():name(),      
      loss = self:loss(),
      sampler = self._sampler:report(),
      epoch_duration = self._epoch_duration,
      batch_duration = self._batch_duration,
      example_speed = self._example_speed,
      num_batches = self._num_batches,
      batch_speed = self._batch_speed
   }
   if self._feedback then
      report.feedback = self._feedback:report()
   end
   if self._visitor then
      report.visitor = self._visitor:report()
   end
   return report
end

function Propagator:setSampler(sampler)
   if self._sampler then
      -- initialize sampler with old sampler values without overwrite
      sampler:setup{batch_size=self._sampler:batchSize(), overwrite=false}
   end
   self._sampler = sampler
end

function Propagator:sampler()
   return self._sampler
end

function Propagator:setObserver(observer)
   if not torch.typename(observer) and type(observer) == 'table' then
      --if list, make composite observer
      observer = CompositeObserver(observer)
   end
   self._observer = observer
end

function Propagator:observer()
   return self._observer
end

function Propagator:id()
   return self._id
end

function Propagator:setModel(model)
   self._model = model
end

function Propagator:model()
   return self._model
end

function Propagator:setVisitor(visitor)
   if not torch.typename(visitor) and type(visitor) == 'table' then
      --if list, make visitor_chain
      visitor = VisitorChain{visitors=visitor}
   end
   self._visitor = visitor
end



function Propagator:visitor()
   return self._visitor
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

function Propagator:updateLoss(batch)
   self._loss = (
                     ( self._samples_seen * self._loss ) 
                     + 
                     ( batch:nSample() * batch:loss() )
                ) / self._samples_seen + batch:nSample()
                
   self._samples_seen = self._samples_seen + batch:nSample()
end

function Propagator:resetLoss()
   self._loss = 0
   self._samples_seen = 0
end

function Propagator:loss()
   return self._loss
end
