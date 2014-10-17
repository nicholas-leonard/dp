require 'dp'
require 'cutorch'
require 'cunn'
require 'cunnx'

local dptest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 10000
local times = {}
local dptestx = {}

function dptest.languagemodel()
   local ds = dp.BillionWords{train_file='train_tiny.th7',context_size=5}
   local hierarchy = ds:hierarchy()
   local train = ds:trainSet()
   local nn_inputs = {}
   local a = torch.Timer()
   for i=1,nloop,512 do
      local batch = train:sub(i,i+511)
      local targets = batch:targets():forward('b')
      table.insert(nn_inputs, {batch:inputs():forward('bt'), targets})
   end
   print("input Time ".. a:time().real)
   
   local tm = {}
   local title = 'language model forward'
   times[title] = tm
   
   local model = dp.Sequential{
      models = {
         dp.Dictionary{
            dict_size = ds:vocabularySize(),
            output_size = 50
         },
         dp.Neural{
            input_size = 50*5, 
            output_size = 50, 
            transfer = nn.Tanh()
         },
         dp.SoftmaxTree{
            input_size = 50, 
            hierarchy = hierarchy,
            root_id = 880542
         }
      }
   }
   model:zeroStatistics()
   
   a:reset()
   local resdp, batch
   for i=1,nloop,512 do
      batch = train:sub(batch, i, i+511)
      local carry = batch:carry()
      carry:putObj('nSample', 512)
      resdp = model:forward(batch:inputs(), carry)
   end
   tm.dp = a:time().real
   print("dp Time ".. a:time().real)--]]
   
   local tm4 = {}
   local title = 'language model forward cuda'
   times[title] = tm4
   
   model:cuda()
   model:zeroStatistics()
   
   a:reset()
   local resdpCuda
   for i=1,nloop,512 do
      batch = train:sub(batch, i, i+511)
      local carry = batch:carry()
      carry:putObj('nSample', 512)
      resdpCuda = model:forward(batch:inputs(), carry)
   end
   tm4.dp = a:time().real
   tm4.nn = tm.dp
   print("dp cuda Time ".. a:time().real)
   
   
   local trunk = nn.Sequential()
   trunk:add(nn.LookupTable(ds:vocabularySize(),50))
   trunk:add(nn.Reshape(50*5))
   trunk:add(nn.Linear(50*5,50))
   trunk:add(nn.Tanh())
   
   local para = nn.ParallelTable()
   para:add(trunk)
   para:add(nn.Identity())
   local mlp = nn.Sequential()
   mlp:add(para)
   mlp:add(nn.SoftMaxTree(50,hierarchy,880542))
   local groundtruth = mlp:forward(nn_inputs[1])
   a:reset()
   for i = 1,#nn_inputs do
      groundtruth = mlp:forward(nn_inputs[i])
   end
   tm.nn = a:time().real
   print("nn Time ".. a:time().real)
   
   local tm2 = {}
   local title = 'language model softmax forward'
   times[title] = tm2
    
   mlp = nn.Sequential()
   mlp:add(trunk)
   mlp:add(nn.Linear(50,800000))
   mlp:add(nn.LogSoftMax())
   
   tm2.dp = tm.nn
   local groundtruth = mlp:forward(nn_inputs[1][1])
   a:reset()
   for i = 1,#nn_inputs do
      groundtruth = mlp:forward(nn_inputs[i][1])
   end
   tm2.nn = a:time().real
   print("softmax Time ".. a:time().real)
   
   local tm3 = {}
   local title = 'language model softmax focused forward'
   times[title] = tm3
   
   mlp = nn.Sequential()
   mlp:add(trunk)
   mlp:add(nn.Linear(50,100))
   mlp:add(nn.LogSoftMax())
   
   tm3.dp = tm.nn
   local groundtruth = mlp:forward(nn_inputs[1][1])
   a:reset()
   for i = 1,#nn_inputs do
      groundtruth = mlp:forward(nn_inputs[i][1])
   end
   tm3.nn = a:time().real
   print("softmax focused Time ".. a:time().real)
end

function nn.testBenchmark(tests)
   math.randomseed(os.time())
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.nn / (tm.dp or 1e6)))
   end
end

nn.testBenchmark()
