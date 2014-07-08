------------------------------------------------------------------------
--[[ SoftmaxTree ]]--
-- A hierarchy of softmaxes.
-- Used for computing the likelihood of a leaf class.
-- Use with TreeNLL Loss.
-- Requires a tensor mapping parent_ids to child_ids. 
-- Root_id defaults to 1
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
      {arg='root_id', type='number | string', default=1,
       help='id of the root of the tree.'},
      {arg='typename', type='string', default='softmaxtree', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   require 'nnx'
   self._module = nn.SoftMaxTree(self._input_size, hierarchy, root_id, false)
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
   if carry.evaluate then 
      self._module:evaluate()
   else
      self._module:training()
   end
   if not (carry.targets and carry.targets.isClassView) then
      error"carry.targets should refer to a ClassView of targets"
   end
   local targets = carry.targets:forward('b', self._target_type)
   -- outputs a column vector of likelihoods of targets
   self:outputAct(self._module:forward{self:inputAct(), targets})
   return carry
end

function SoftmaxTree:_backward(carry)
   if not (carry.targets and carry.targets.isClassView) then
      error"carry.targets should refer to a ClassView of targets"
   end
   local targets = carry.targets:forward('b', self._target_type)
   local input_grad
   if self._acc_update then 
      input_grad = self._module:updateGradInput({self:inputAct(), targets}, self:outputGrad())
   else
      input_grad = self._module:backward({self:inputAct(), targets}, self:outputGrad(), self._acc_scale)
   end
   self:inputGrad(input_grad)
   return carry
end

function SoftmaxTree:_type(type)
   self._input_type = type
   self._output_type = type
   if type == 'torch.CudaTensor' then
      require 'cunnx'
      self._target_type = 'torch.CudaTensor'
   else
      self._target_type = 'torch.IntTensor'
   end
   self._module:type(type)
   return self
end

function SoftmaxTree:zeroGradParameters()
   if not self._acc_update then
      self._module:zeroGradParameters(true)
   end
end

-- if after feedforward, returns active parameters 
-- else returns all parameters
function SoftmaxTree:parameters()
   return self._module:parameters(true)
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
   clone._module = self._module:sharedClone()
   return clone
end

