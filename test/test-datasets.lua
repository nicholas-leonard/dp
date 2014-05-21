-- we keep these tests appart since they may require downloads and 
-- require lots of memory
local mytester 
local dptest = {}
local mediator = dp.Mediator()

function dptest.billionwords()
   local ds = dp.BillionWords{train_file='train_tiny.th7', context_size=10}
   local hierarchy= ds:hierarchy()
   mytester:assert(torch.type(hierarchy) == 'table')
   local root = hierarchy[-1]
   mytester:assert(torch.type(root) == 'torch.IntTensor')
   -- dp
   local batch = ds:sub(1, 100)
   local input = ds:inputs()
   local target = dp.targets()
   local model = dp.SoftmaxTree{input_size=10, hierarchy=hierarchy}
   -- forward backward
   local act, carry = model:forward(input, batch:carry())
   local gradWeight = model:parameters().weight1.grad:clone()
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assertTableEq(act:feature():size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(grad:feature():size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local gradWeight2 = model:parameters().weight1.grad:clone()
   mytester:assertTensorNe(gradWeight, gradWeight2, 0.00001)
   -- share
   local model2 = model:sharedClone()   
   -- update
   local weight = model:parameters().weight1.param:clone()
   local act_ten = act:feature():clone()
   local grad_ten = grad:feature():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   model:accept(visitor)
   local weight2 = model:parameters().weight1.param:clone()
   mytester:assertTensorNe(weight, weight2, 0.00001)
   model:doneBatch()
   -- forward backward
   local act2, carry2 = model2:forward(input, {nSample=5, targets=target})
   local grad2, carry2 = model2:backward(dp.DataTensor{data=grad_tensor}, carry2)
   mytester:assertTensorNe(act_ten, act2:feature(), 0.00001)
   mytester:assertTensorNe(grad_ten, grad2:feature(), 0.00001)
   local act, carry = model:forward(input, {nSample=5, targets=target})
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assertTensorEq(act:feature(), act2:feature(), 0.00001)
   mytester:assertTensorEq(grad:feature(), grad2:feature(), 0.00001)
end

function dp.testDatasets(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

