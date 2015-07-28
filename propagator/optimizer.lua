------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Propagator subclass
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------
local Optimizer, parent = torch.class("dp.Optimizer", "dp.Propagator")
Optimizer.isOptimizer = true

function Optimizer:__init(config)
   config = config or {}
   local args, loss, sampler, acc_update, callback, update_interval, 
      stats = xlua.unpack(
      {config},
      'Optimizer', 
      'Optimizes a model on a training dataset',
      {arg='loss', type='nn.Criterion', req=true,
       help='a neural network Criterion to evaluate or minimize'},
      {arg='sampler', type='dp.Sampler', 
       help='used to iterate through the train set. ' ..
       'Defaults to dp.ShuffleSampler()'},
      {arg='acc_update', type='boolean', default=false,
       help='when true, uses the faster accUpdateGradParameters, '..
       'which performs an inplace update (no need for param gradients). '..
       'However, this also means that Momentum, WeightDecay and other '..
       'such gradient modifying Visitors cannot be used.'},
      {arg='callback', type='function', req=true,
       help='function(model, report) that does things like'..
       'update model, gather statistics, decay learning rate, etc.'},
      {arg='update_interval', type='number', default=1,
       help='update the model every update_interval'},
      {arg='stats', type='boolean', default=true,
       help='display statistics'}
   )
   self._update_interval = update_interval
   self._acc_update = acc_update
   config.loss = loss
   config.callback = callback
   config.sampler = sampler or dp.ShuffleSampler()
   config.stats = stats
   parent.__init(self, config)
end

function Optimizer:setup(config)
   parent.setup(self, config)
   self._model:zeroGradParameters() -- don't forget this, else NaN errors
end
      
function Optimizer:propagateBatch(batch, report)
   self._model:training()
   self:forward(batch)
   self:monitor(batch, report)
   self:backward(batch)
   if report.epoch % self._update_interval == 0 then
      self._callback(self._model, report)
   end
   self:doneBatch(report)
end

function Optimizer:backward(batch)
   local input = batch:inputs():input()
   local target = batch:targets():input()
   target = self._target_module:forward(target)
   -- estimate gradient of loss w.r.t. outputs
   self.gradOutput = self._loss:backward(self.output, target)
   -- backprop through model
   if self._include_target then
      input = {input, target}
   end
   if self._acc_update then 
      self.gradInput = self._model:updateGradInput(input, self.gradOutput)
   else
      self.gradInput = self._model:backward(input, self.gradOutput)
   end
   -- so that visitors can known whether or not gradParams were updated
   self._model.dpnn_accGradParameters = not self._acc_update
end
