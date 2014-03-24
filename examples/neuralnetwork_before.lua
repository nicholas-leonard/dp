require 'torch'
require 'sys'
require 'optim'
require 'nn'
require 'paths'

require 'equanimity/optim/ConfusionMatrix'
require 'DataSet/mnist'

--[[parse command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST MLP Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-numHidden', 200, 'number of hidden units')
cmd:option('-batchSize', 32, 'number of examples per batch')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:text()
opt = cmd:parse(arg or {})

--[[Performance Metrics]]--
-- classes
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix2(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   torch.setdefaulttensortype('torch.CudaTensor')
end

--[[Load DataSets]]--
train_data = DataSet.Mnist{which_set='train'}
valid_data = DataSet.Mnist{which_set='valid'}
test_data = DataSet.Mnist{which_set='test'}

ninputs = train_data:n_dimensions()
nhidden = opt.numHidden
noutputs = 10


--[[Build a neural network]]--
model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhidden))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden,noutputs))
model:add(nn.LogSoftMax())

--[[Add a loss function]]--
criterion = nn.ClassNLLCriterion()

--[[Build a training algorithm]]--
optimState = {
      learningRate = opt.learningRate,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters, gradParameters = model:getParameters()
   
--TODO: make this into a class:
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   shuffle = torch.randperm(train_data:size())

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for start = 1,train_data:size(),opt.batchSize do
      local stop = math.min(start+opt.batchSize-1,train_data:size())
      -- disp progress
      xlua.progress(start, train_data:size())

      -- create mini batch
      local indices = shuffle:sub(start,stop):long()
      local inputs = train_data:inputs():index(1, indices)
      local targets = train_data:targets():index(1, indices)

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

   -- time taken
   time = sys.clock() - time
   time = time / train_data:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   
   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
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

train()

--[[

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end]]--
