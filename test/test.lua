local mytester 
local dptest = {}
local msize = 100

function dptest.uid()
   local uid1 = dp.uniqueID()
   mytester:asserteq(type(uid1), 'string', 'type(uid1) == string')
   local uid2 = dp.uniqueID()
   local uid3 = dp.uniqueID('mynamespace')
   mytester:assertne(uid1, uid2, 'uid1 ~= uid2')
   mytester:assertne(uid2, uid3, 'uid2 ~= uid3')
end
function dptest.datatensor()
   
end
function dptest.gcn_zero_vector()
   -- Global Contrast Normalization
   -- Test that passing in the zero vector does not result in
   -- a divide by 0 error
   local dataset = dp.DataSet{
      which_set='train', inputs=torch.zeros(1, 1),
      axes={'b','f'}, sizes={1,1}
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
      which_set='train', axes={'b','f'}, sizes={1,1},
      inputs=torch.rand(3,9)
   }
   
   local preprocess = dp.GCN{std_bias=0.0, use_std=false}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs(1):data()
   local norms = torch.pow(result, 2):sum(2):sqrt()
   local max_norm_error = torch.abs(norms:add(-1)):max()
   mytester:assert(max_norm_error < 3e-5)
end

function dp.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)
   return mytester
end

