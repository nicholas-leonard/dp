------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a module in order to evaluate its criterion, train, etc.
-- To make your own training algorithm, you can build your own 
-- propagator. If you can reduce it to reusable components, you could 
-- then refactor these into visitors, observers, etc.
------------------------------------------------------------------------
local Propagator = torch.class("dp.Propagator")
Propagator.isPropagator = true

function Propagator:__init(config)   
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, loss, visitor, sampler, observer, feedback, progress,
      verbose, stats = xlua.unpack(
      {config},
      'Propagator', 
      'Propagates Batches sampled from a DataSet using a Sampler '..
      'through a Model in order to evaluate a Loss, provide Feedback '.. 
      'or train the model',
      {arg='loss', type='nn.Criterion', req=true,
       help='a neural network Criterion to evaluate or minimize'},
      {arg='visitor', type='dp.Visitor',
       help='visits models at the end of each batch propagation to '.. 
       'perform parameter updates and/or gather statistics, etc.'},
      {arg='sampler', type='dp.Sampler', 
       help='Iterates through a DataSet. [Default=dp.Sampler()]'},
      {arg='observer', type='dp.Observer', 
       help='observer that is informed when an event occurs.'},
      {arg='feedback', type='dp.Feedback',
       help='takes predictions, targets, model and visitor as input '..
       'and provides feedback through report(), setState, or mediator'},
      {arg='progress', type='boolean', default=false, 
       help='display progress bar'},
      {arg='verbose', type='boolean', default=true,
       help='print verbose information'},
      {arg='stats', type='boolean', default=false,
       help='display performance statistics (speed, etc). '..
      'Only applies if verbose is true.'}
   )
   self:sampler(sampler or dp.Sampler())
   self:loss(loss)
   self:observer(observer)
   self:feedback(feedback)
   self:visitor(visitor)
   self._progress = progress
   self._verbose = verbose
   self._stats = stats
   self.output = {}
end

function Propagator:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, id, model, mediator, mem_type, dataset, overwrite
      = xlua.unpack(
      {config},
      'Propagator:setup', 
      'Post-initialization setup of the Propagator',
      {arg='id', type='dp.ObjectID', req=true,
       help='uniquely identifies the propagator.'},
      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},
      {arg='mediator', type='dp.Mediator', req=true,
       help='used for inter-object communication.'},
      {arg='dataset', type='dp.DataSet', 
       help='This might be useful to determine the type of targets. ' ..
       'Propagator should not hold a reference to a dataset due to ' ..
       'the propagator\'s possible serialization.'}
   )
   assert(torch.isTypeOf(id, 'dp.ObjectID'))
   self._id = id
   assert(torch.isTypeOf(mediator, 'dp.Mediator'))
   self._mediator = mediator
   self:model(model)
   self._sampler:setup{mediator=mediator, model=model}
   if self._observer then self._observer:setup{
      mediator=mediator, subject=self
   } end
   if self._feedback then self._feedback:setup{
      mediator=mediator, propagator=self, dataset=dataset
   } end
   if self._visitor then self._visitor:setup{
      mediator=mediator, model=self._model, propagator=self
   } end
end

function Propagator:propagateEpoch(dataset, report)
   if self._feedback then
      self._feedback:reset()
   end
   
   -- local vars
   local start_time = sys.clock()
   local batch, i, n, last_n
   local n_batch = 1
   
   if self._stats and self._verbose then
      print('==> epoch # '..(report.epoch + 1)..' for '..self:name()..' :')
   end
   
   local sampler = self._sampler:sampleEpoch(dataset)
   while true do
      -- reuse the batch object
      if batch then
         assert(torch.type(batch) == 'dp.Batch')
      end
      
      batch, i, n = sampler(batch)
      if not batch then 
         -- for aesthetics :
         if self._progress then
            xlua.progress(last_n, last_n)
         end
         break 
      end
      
      self:propagateBatch(batch, report)
      
      if self._progress then
         -- display progress
         xlua.progress(i, n)
      end
      last_n = n
      n_batch = n_batch + 1
   end
   
   -- time taken
   self._epoch_duration = sys.clock() - start_time
   self._batch_duration = self._epoch_duration / last_n
   self._example_speed = last_n / self._epoch_duration
   self._batch_speed = (n_batch / self._epoch_duration)
   if self._stats and self._verbose then
      print("==> example speed = "..self._example_speed..' examples/s')
   end
end      

function Propagator:propagateBatch(batch)
   error"NotImplementedError"
end

function Propagator:monitor(batch, report)
   -- monitor error and such
   if self._feedback then
      self._feedback:add(batch, self.output, report)
   end
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneFeedback", report, batch)
   return carry
end

function Propagator:doneBatch(report)
   -- zero gradients, statistics, etc.
   self._model:doneBatch()
   self._loss:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneBatch", report)
   self.output = {}
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
      loss = self._loss:report(), error"TODO"
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

function Propagator:sampler(sampler)
   if sampler then
      self._sampler = sampler
      return
   end
   return self._sampler
end

function Propagator:id(id)
   if self._id then
      self._id = id
      return
   end
   return self._id
end

function Propagator:name()
   return self._id:name()
end

function Propagator:model(model)
   if model then
      self._model = model
      return
   end
   return self._model
end

function Propagator:observer(observer)
   if observer then
      if torch.type(observer) == 'table' then
         --if list, make composite observer
         observer = dp.CompositeObserver(observer)
      end
      self._observer = observer
      return
   end
   return self._observer
end

function Propagator:visitor(visitor)
   if visitor then
      if torch.type(visitor) == 'table' then
         --if list, make visitor_chain
         visitor = dp.VisitorChain{visitors=visitor}
      end
      self._visitor = visitor
      return
   end
   return self._visitor
end

function Propagator:feedback(feedback)
   if feedback then
      if torch.type(feedback) == 'table' then
         --if list, make visitor_chain
         feedback = dp.CompositeFeedback{feedbacks=feedback}
      end
      self._feedback = feedback
      return
   end
   return self._feedback
end

function Propagator:loss(loss)
   if loss then
      assert(torch.isTypeOf(loss, 'nn.Criterion'), "Expecting nn.Criterion instance")
      self._loss = loss
      return
   end
   return self._loss
end
function Propagator:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
   if self._loss then
      self._loss:verbose(self._verbose)
   end
   if self._feedback then
      self._feedback:verbose(self._verbose)
   end
   if self._visitor then 
      self._visitor:verbose(self._verbose)
   end
   if self._observer then
      self._observer:verbose(self._verbose)
   end
   --TODO self._sampler:verbose()
end

function Propagator:silent()
   self:verbose(false)
end

function Propagator:type(new_type)
   if self._loss then
      self._loss:type(new_type)
   end
end

function Propagator:float()
   return self:type('torch.FloatTensor')
end

function Propagator:double()
   return self:type('torch.DoubleTensor')
end

function Propagator:cuda()
   return self:type('torch.CudaTensor')
end

function Propagator:int()
   return self:type('torch.IntTensor')
end

function Propagator:long()
   return self:type('torch.LongTensor')
end
