local mytester 
local dptest = {}

function dptest.uid()
   local uid1 = dp.uniqueID()
   mytester:asserteq(type(uid1), 'string', 'type(uid1) == string')
   local uid2 = dp.uniqueID()
   local uid3 = dp.uniqueID('mynamespace')
   mytester:assertne(uid1, uid2, 'uid1 ~= uid2')
   mytester:assertne(uid2, uid3, 'uid2 ~= uid3')
end
function dptest.datatensor()
   local data = torch.rand(3,4)
   local axes = {'b','f'}
   local sizes = {3, 4}
   local d = dp.DataTensor{data=data, axes=axes, sizes=sizes}
   local t = d:feature()
   function test() return d:image() end
   mytester:asserteq(t:dim(),2)
   mytester:assert(not pcall(test))
   
end
function dptest.imagetensor()
   local size = {3,32,32,3}
   local feature_size = {3,32*32*3}
   local data = torch.rand(unpack(size))
   local axes = {'b','h','w','c'}
   local d = dp.ImageTensor{data=data, axes=axes}
   -- convert to image (shouldn't change anything)
   local i = d:image()
   mytester:assertTensorEq(i, data, 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- convert to feature (should colapse last dims)
   local t = d:feature()
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assertTensorEq(i, data, 0.00001)
   -- convert to image (should expand last dim)
   local i = d:image()
   mytester:assertTensorEq(i, data, 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
end
function dptest.classtensor()
   local size = {48,4}
   local data = torch.rand(unpack(size))
   local axes = {'b','t'}
   local d = dp.ClassTensor{data=data, axes=axes}
   local t = d:multiclass()
   mytester:assertTableEq(t:size():totable(), size, 0.0001)
   mytester:assertTensorEq(t, data, 0.00001)
   local i = d:class()
   mytester:asserteq(i:dim(),1)
   mytester:assertTableEq(i:size(1), size[1], 0.0001)
   mytester:assertTensorEq(i, data:select(2,1), 0.00001)
end
function dptest.gcn_zero_vector()
   -- Global Contrast Normalization
   -- Test that passing in the zero vector does not result in
   -- a divide by 0 error
   local dataset = dp.DataSet{
      which_set='train', inputs=dp.DataTensor{data=torch.zeros(1, 1)}
   }

   --std_bias = 0.0 is the only value for which there 
   --should be a risk of failure occurring
   local preprocess = dp.GCN{sqrt_bias=0.0, use_std=true}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs(1):data():sum()

   mytester:assert(not _.isNaN(result))
   mytester:assert(_.isFinite(result))
end
function dptest.gcn_unit_norm()
   -- Global Contrast Normalization
   -- Test that using std_bias = 0.0 and use_norm = True
   -- results in vectors having unit norm

   local dataset = dp.DataSet{
      which_set='train', inputs=dp.DataTensor{data=torch.rand(3,9)}
   }
   
   local preprocess = dp.GCN{std_bias=0.0, use_std=false}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs(1):data()
   local norms = torch.pow(result, 2):sum(2):sqrt()
   local max_norm_error = torch.abs(norms:add(-1)):max()
   mytester:assert(max_norm_error < 3e-5)
end
function dptest.zca()
   -- Confirm that ZCA.inv_P_ is the correct inverse of ZCA._P.
   local dataset = dp.DataSet{
      which_set='train', inputs=dp.DataTensor{data=torch.randn(15,10)}
   }
   local preprocess = dp.ZCA()
   preprocess._unit_test = true
   dataset:preprocess{input_preprocess=preprocess}
   local function is_identity(matrix)
      local identity = torch.eye(matrix:size(1))
      local abs_diff = torch.abs(identity:add(-matrix))
      return (torch.lt(abs_diff,.00001):int():min() ~= 0)
   end
   local data = dataset:inputs(1):data()
   mytester:assert(table.eq(preprocess._P:size():totable(),{data:size(2),data:size(2)}))
   mytester:assert(not is_identity(preprocess._P))
   mytester:assert(is_identity(preprocess._P*preprocess._inv_P))
end
function dptest.neural()
   local tensor = torch.randn(5,10)
   -- dp
   local input = dp.DataTensor{data=tensor}
   local layer = dp.Neural{input_size=10, output_size=2, transfer=nn.Tanh()}
   local act, carry = layer:forward(input, {nSample=5})
   local grad = layer:backward(input, carry)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.Linear(10,2)
   m:share(layer._affine, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   local mlp_act = mlp:forward(tensor)
   local mlp_grad = mlp:backward(tensor, tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:feature(), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:feature(), 0.00001)
end

function dp.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

