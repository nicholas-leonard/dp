------------------------------------------------------------------------
--[[ SoftmaxForest ]]--
-- Model, Composite
-- A mixture of n SoftMaxTree experts that uses an MLP gater.
------------------------------------------------------------------------
local SoftmaxForest, parent = torch.class("dp.SoftmaxForest", "dp.SoftmaxTree")
SoftmaxForest.isSoftmaxForest = true

function SoftmaxForest:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, hierarchy, root_id, gater_size, gater_act,
      typename = xlua.unpack(
      {config},
      'SoftmaxForest', 
      'A mixture of softmax trees',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='hierarchy', type='table', req=true,
       help='A list of tables mapping parent_ids to a tensor of child_ids'},
      {arg='root_id', type='table', req=true,
       help='a list of ids of the root of the trees.'},
      {arg='gater_size', type='table', default='',
       help='a list of hidden layer sizes for the gater'},
      {arg='gater_act', type='nn.Module', default='',
       help='a list of hidden transfer functions. Default=nn.Tanh()'},
      {arg='typename', type='string', default='softmaxforest', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._gater_act = (gater_act == '') and nn.Tanh() or gater_act
   self._gater_size = (gater_size == '') and {} or gater_size
   
   -- experts
   self._experts = nn.ConcatTable()
   self._smts = {}
   for i,tree in ipairs(hierarchy) do
      local smt = nn.SoftMaxTree(self._input_size, tree, root_id[i], config.acc_update)
      table.insert(self._smts, smt)
      self._experts:add(smt)
   end
   -- gater
   self._gater = nn.Sequential()
   self._gater:add(nn.SelectTable(1)) -- ignore targets
   local inputSize = self._input_size
   for i,hiddenSize in ipairs(self._gater_size) do 
      self._gater:add(nn.Linear(inputSize, hiddenSize))
      self._gater:add(self._gater_act:clone())
      inputSize = hiddenSize
   end
   self._gater:add(nn.Linear(inputSize, self._experts:size()))
   self._gater:add(nn.SoftMax())
   -- mixture
   self._trunk = nn.ConcatTable()
   self._trunk:add(self._gater)
   self._trunk:add(self._experts)
   self._mixture = nn.MixtureTable()
   self._module = nn.Sequential()
   self._module:add(self._trunk)
   self._module:add(self._mixture)
   
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'b'
   config.tags = config.tags or {}
   dp.Layer.__init(self, config)
   self._target_type = 'torch.IntTensor'
end


function SoftmaxForest:zeroGradParameters()
   if not self._acc_update then
      for i,smt in ipairs(self._experts) do
         smt:zeroGradParameters(true)
      end
      self._gater:zeroGradParameters()      
   end
end

function SoftmaxForest:parameters()
   local param, gradParam = {}, {}
   local paramG, gradParamG = self._gater:parameters()
   _.push(param, unpack(paramG))
   _.push(gradParam, unpack(gradParamG))
   local n = #param
   for i,smt in ipairs(self._smts) do
      local paramE, gradParamE = smt:parameters(true)
      for k,p in pairs(paramE) do
         param[n+k] = p
         gradParam[n+k] = gradParamE[k]
      end
      n = n+(smt.maxParentId*2)
   end
   return param, gradParam
end

function SoftmaxForest:sharedClone()
   error"Not Implemented"
end

function SoftmaxForest:maxNorm(max_out_norm, max_in_norm)
   for i,smt in ipairs(self._smts) do 
      smt:maxNorm(max_out_norm, true)
      if self._acc_update then
         smt.updates = {}
      end
   end
   local params = self._gater:parameters()
   for i,param in ipairs(params) do
      if param:dim() == 2 then
         if max_out_norm then
            param:renorm(1, 2, max_out_norm)
         end
         if max_in_norm then
            param:renorm(2, 2, max_in_norm)
         end
      end
   end
end

function SoftmaxForest:pushDropout(dropout)
   local para = nn.ParallelTable()
   para:add(dropout)
   para:add(nn.Identity())
   --TODO add dropout to hidden layers of gater
   self._module:insert(para, 1)
end
