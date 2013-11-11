require 'xlua'
require 'sys'
require 'torch'

------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a model in order to evaluate its criteria, train the model,
-- etc.
------------------------------------------------------------------------

local Propagator = torch.class("dp.Propagator")

function Propagator:__init(...)   

   local args, dataset, model, batch_size, mem_type, logger, plot, progress
      = xlua.unpack(
      {... or {}},
      'Propagator', nil,
      {arg='model', type='nn.Module', req=true,
       help='the model that is to be trained or tested',},
      {arg='sampler', type='dp.Sampler', default=dp.Sampler(),
       help='used to iterate through the train set'},
      {arg='dataset', type='dp.Dataset',
       help='used to setup sampler if not done already'},
      {arg='logger', type='dp.Logger',
       help='an object implementing the Logger interface'},
      {arg='criterion', type='nn.Criterion', req=true,
       help='a neural network criterion to evaluate or minimize'},
      {arg='mem_type', type='string', default='float',
       help='double | float | cuda'},
      {arg='plot', type='boolean', default=false,
       help='live plot of confusion matrix'},
      {arg='progress', type'boolean', default=true, 
       help='display progress bar'}
   )
   self._epoch = 1
   self._dataset = dataset
   self._model = model
   self._batch_size = batch_size
   self._mem_type = mem_type
   self._logger = logger
   self._plot = plot
   self._progress = progress
end

function Propagator:setup(...)
   local dataset
end

function Propagator:doEpoch()
   -- local vars
   local time = sys.clock()
   -- do one epoch
   print('==> doing epoch on training data:')
   print('==> online epoch # ' .. self._epoch .. 
         ' [batchSize = ' .. self._batch_size .. ']')
         
   for inputs, targets in self:sampler() do
      if self._progress then
         -- disp progress
         xlua.progress(start, self.DataSet:size())
      end
      self:doBatch(inputs, targets)
   end
   -- time taken
   time = sys.clock() - time
   time = time / self.DataSet:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   
   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if self._plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end      

      
function Propagator:doBatch(inputs, targets)

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
end

------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------

local Optimizer = torch.class("dp.Optimizer", "dp.Propagator")

function Optimizer:__init(...)
   args, self.learning_rate, self.weight_decay, self.momentum, self.shuffle
      = xlua.unpack(
      {... or {}},
      'Optimizer', 
      'Trains a model on a training dataset',
      {arg='sampler', type='dp.Sampler', default=dp.ShuffleSampler(),
       help='used to iterate through the train set'},
      {arg='learning_rate', type='number', req=true,
       help='learning rate at start of learning'}
      {arg='weight_decay', type='number', default=0 
       help='weight decay coefficient'},
      {arg='momentum', type='number', default=0,
       help='momentum of the parameter gradients'}
   )
   Propagator.__init(self, args)
end

function Optimizer:doEpoch(...)

end

------------------------------------------------------------------------
--[[ Evaluator ]]--
-- Tests (evaluates) a model using a sampling distribution.
-- For evaluating the generalization of the model, seperate the 
-- training data from the test data. The Evaluator can also be used 
-- for early-stoping.
------------------------------------------------------------------------


local Evaluator = torch.class("dp.Evaluator", "dp.Propagator")

function Evaluator:__init(...)

end

function Evaluator:doEpoch(...)

end
