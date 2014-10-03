------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a model in order to evaluate its Loss, train the model, etc.
-- To make your own training algorithm, you can build your own 
-- propagator. If you can reduce it to reusable components, you could 
-- then refactor these into visitors, observers, etc.
------------------------------------------------------------------------
local Propagator = torch.class("dp.Propagator")
Propagator.isPropagator = true

function Propagator:__init(config)   
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, loss, visitor, sampler, observer, feedback, progress, stats
      = xlua.unpack(
      {config},
      'Propagator', 
      'Propagates Batches sampled from a DataSet using a Sampler '..
      'through a Model in order to evaluate a Loss, provide Feedback '.. 
      'or train the model',
      {arg='loss', type='dp.Loss | nn.Criterion', req=true,
       help='a neural network Loss to evaluate or minimize'},
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
      {arg='stats', type='boolean', default=false,
       help='display statistics'}
   )
   self:setSampler(sampler or dp.Sampler())
   self:setLoss(loss)
   self:setObserver(observer)
   self:setFeedback(feedback)
   self:setVisitor(visitor)
   self._progress = progress
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
   assert(id.isObjectID)
   self._id = id
   assert(mediator.isMediator)
   self._mediator = mediator
   self:setModel(model)
   self._sampler:setup{mediator=mediator, model=model}
   self._loss:setup{mediator=mediator, id=id:create("loss")}
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
   
   if self._stats then
      print('==> epoch # '..(report.epoch + 1)..' for '..self:name())
   end
   
   local sampler = self._sampler:sampleEpoch(dataset)
   while true do
      -- reuse the batch object
      batch, i, n = sampler(batch)
      if (not batch) and self._progress then 
         -- for aesthetics :
         xlua.progress(last_n, last_n)
         break 
      end
      self:propagateBatch(batch, report)
      if self._progress then
         -- disp progress
         xlua.progress(i, n)
      end
      last_n = n
      n_batch = n_batch + 1
   end
   
   if self._progress and not self._stats then
      print"\n"
   end
   
   -- time taken
   self._epoch_duration = sys.clock() - start_time
   self._batch_duration = self._epoch_duration / last_n
   self._example_speed = last_n / self._epoch_duration
   self._num_batches = last_n / n_batch
   self._batch_speed = (self._num_batches / self._epoch_duration)
   if self._stats then
      print("\n==> epoch size = "..last_n..' examples')
      print("==> batch duration = "..(self._batch_duration*1000)..' ms')
      print("==> epoch duration = " ..self._epoch_duration..' s')
      print("==> example speed = "..self._example_speed..' examples/s')
      print("==> batch speed = "..self._batch_speed..' batches/s')
   end
end      

function Propagator:propagateBatch(batch)
   error"NotImplementedError"
end


function Propagator:monitor(batch, report, carry)
   -- monitor error and such
   if self._feedback then
      self._feedback:add(batch, self.output, carry, report)
   end
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneFeedback", report, batch)
   return carry
end

function Propagator:doneBatch(report, carry)
   -- zero gradients, statistics, etc.
   self._model:doneBatch()
   self._loss:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneBatch", report, carry)
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
      loss = self._loss:report(),
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
   self._sampler = sampler
end

function Propagator:sampler()
   return self._sampler
end

function Propagator:id()
   return self._id
end

function Propagator:name()
   return self._id:name()
end

function Propagator:setModel(model)
   self._model = model
end

function Propagator:model()
   return self._model
end

function Propagator:setObserver(observer)
   if not torch.typename(observer) and type(observer) == 'table' then
      --if list, make composite observer
      observer = dp.CompositeObserver(observer)
   end
   self._observer = observer
end

function Propagator:observer()
   return self._observer
end

function Propagator:setVisitor(visitor)
   if not torch.typename(visitor) and type(visitor) == 'table' then
      --if list, make visitor_chain
      visitor = dp.VisitorChain{visitors=visitor}
   end
   self._visitor = visitor
end

function Propagator:visitor()
   return self._visitor
end

function Propagator:setFeedback(feedback)
   if not torch.typename(feedback) and type(feedback) == 'table' then
      --if list, make visitor_chain
      feedback = dp.CompositeFeedback{feedbacks=feedback}
   end
   self._feedback = feedback
end

function Propagator:feedback()
   return self._feedback
end

function Propagator:setLoss(loss)
   if not loss.isLoss then
      print("Propagator:setLoss Warning : "..
         "'loss' argumetn isn't an instance of dp.Loss."..
         "Assuming it's a nn.Criterion instance."..
         "Wrapping it in dp.Criterion (this doesn't always work as-is)"
      )
      loss = dp.Criterion{criterion=loss}
   end
   self._loss = loss
end

function Propagator:loss()
   return self._loss
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
