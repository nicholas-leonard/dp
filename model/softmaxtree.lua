------------------------------------------------------------------------
--[[ SoftmaxTree ]]--
-- A hierarchy of softmaxes.
-- Used for computing the likelihood of a leaf class.
-- Use with TreeNLL Loss.
-- Requires a tensor mapping parent_ids to child_ids. 
-- Root_id defaults to 1
-- Should use with acc_update = true for nice speedup
------------------------------------------------------------------------
local SoftmaxTree, parent = torch.class("dp.SoftmaxTree", "dp.Layer")
SoftmaxTree.isSoftmaxTree = true

function SoftmaxTree:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, hierarchy, root_id, typename = xlua.unpack(
      {config},
      'SoftmaxTree', 
      'A hierarchy of softmaxes',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='hierarchy', type='table', req=true,
       help='A table mapping parent_ids to a tensor of child_ids'},
      {arg='root_id', type='number', default=1,
       help='id of the root of the tree.'},
      {arg='typename', type='string', default='softmaxtree', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._module = nn.SoftMaxTree(self._input_size, hierarchy, root_id, config.acc_update)
   self._smt = self._module
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'b'
   config.tags = config.tags or {}
   parent.__init(self, config)
   self._target_type = 'torch.IntTensor'
end

-- requires targets be in carry
function SoftmaxTree:_forward(carry)
   if carry:getObj('evaluate') then 
      self._module:evaluate()
   else
      self._module:training()
   end
   local targetView = carry:getObj('targets')
   if not (targetView and targetView.isClassView) then
      error"expecting a ClassView of targets"
   end
   self._targets = targetView:forward('b', self._target_type)
   -- outputs a column vector of likelihoods of targets
   self:outputAct(self._module:forward{self:inputAct(), self._targets})
   return carry
end

function SoftmaxTree:_backward(carry)
   local input_grad
   if self._acc_update then 
      input_grad = self._module:updateGradInput({self:inputAct(), self._targets}, self:outputGrad())
   else
      input_grad = self._module:backward({self:inputAct(), self._targets}, self:outputGrad(), self._acc_scale)
   end
   self:inputGrad(input_grad[1])
   return carry
end

function SoftmaxTree:updateParameters(lr)
   if self._acc_update then
      self._module:accUpdateGradParameters({self:inputAct(), self._targets}, self:outputGrad(), lr*self._acc_scale)
   else
      self._module:updateParameters(lr)
   end
end

function SoftmaxTree:_type(type)
   self._input_type = type
   self._output_type = type
   if type == 'torch.CudaTensor' then
      self._target_type = 'torch.CudaTensor'
   else
      self._target_type = 'torch.IntTensor'
   end
   self._module:type(type)
   return self
end

-- if after feedforward, returns active parameters 
-- else returns all parameters
function SoftmaxTree:parameters()
   local params, gradParams = self._module:parameters()
   return params, gradParams, nil, self._module.nChildNode * 2
end

function SoftmaxTree:sharedClone()
   local clone = torch.protoClone(self, {
      input_size=self._input_size, hierarchy={[1]=torch.IntTensor{1,2}},
      root_id=1, sparse_init=self._sparse_init,
      typename=self._typename, 
      input_type=self._input_type, output_type=self._output_type,
      module_type=self._module_type, mvstate=self.mvstate
   })
   clone._target_type = self._target_type
   clone._smt = self._smt:sharedClone()
   clone._module = self._smt
   if self._dropout then
      clone:pushDropout(self._dropout:clone())
   end
   return clone
end

function SoftmaxTree:maxNorm(max_out_norm)
   self._smt:maxNorm(max_out_norm, true)
end

function SoftmaxTree:pushDropout(dropout)
   local para = nn.ParallelTable()
   para:add(dropout)
   para:add(nn.Identity())
   local mlp = nn.Sequential()
   mlp:add(para)
   mlp:add(self._module)
   self._module = mlp
end

function SoftmaxTree:_toModule()
   error"Not Implemented : TODO implement using Push/PullTable... (open a ticket to motivate us to do it)"
end
