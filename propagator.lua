require 'xlua'
require 'sys'
require 'torch'
require 'optim'

--[[ TODO ]]--
-- Logger 
-- Specify which values are default setup by Experiment in doc
-- Mediator is referenced by all objects, including observers (setup)
-- Observers own concrete channels


------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a model in order to evaluate its criteria, train the model,
-- etc.
------------------------------------------------------------------------

local Propagator = torch.class("dp.Propagator")

function Propagator:__init(...)   
   local args, model, sampler, logger, criterion,
      visitor, observer, feedback, mem_type, progress
      = xlua.unpack(
      {... or {}},
      'Propagator', nil,
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
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
       'and provides feedback through report(), setState, or mediator'}
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='progress', type'boolean', default=true, 
       help='display progress bar'}
   )
   sampler:setup{batch_size=batch_size, overwrite=false}
   self:setSampler(sampler)
   self:setModel(model)
   self:setMemType(mem_type)
   self:setCriterion(criterion)
   self:setObserver(observer)
   self:setProgress(progress)
   self._observer = observer
   self._feedback = feedback
   self._visitor = visitor
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
      self._observer:setup(self._mediator, self)
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
   return {
      name = self:id():name(),
      sampler = self:sampler:report(),
      feedback = self:feedback():report(),
      visitor = self:visitor:report(),
      model = self:model():report(),
      loss = self:loss(),
      epoch_duration = self._epoch_duration,
      batch_duration = self._batch_duration,
      example_speed = self._example_speed
      num_batches = self._num_batches
      batch_speed = self._batch_speed
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

function Propagate:updateLoss(batch)
   self._loss = (
                     ( self._samples_seen * self._loss ) 
                     + 
                     ( batch:nSample() * batch:loss() )
                ) / self._samples_seen + batch:nSample()
                
   self._samples_seen = self._loss_samples + batch:nSample()
end

function Propagate:resetLoss()
   self._loss = 0
   self._samples_seen = 0
end

function Propagate:loss()
   return self._loss
end
