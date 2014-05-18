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
   self._module = nn.SoftMaxTree(self._input_size, hierarchy, root_id)
   config.typename = typename
   parent.__init(self, config)
end

-- requires targets be in carry
function SoftmaxTree:_forward(carry)
   local activation = self:inputAct()
   if self._dropout then
      -- dropout has a different behavior during evaluation vs training
      self._dropout.train = (not carry.evaluate)
      activation = self._dropout:forward(activation)
      self.mvstate.dropoutAct = activation
   end
   assert(carry.targets and carry.targets.isClassTensor,
      "carry.targets should refer to a ClassTensor of targets")
   local targets = carry.targets:class()
   -- outputs a column vector of likelihoods of targets
   activation = self._module:forward{activation, targets}
   self:outputAct(activation)
   return carry
end

function SoftmaxTree:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local input_act = self.mvstate.dropoutAct or self:inputAct()
   local output_grad = self:outputGrad():select(2,1)
   assert(carry.targets and carry.targets.isClassTensor,
      "carry.targets should refer to a ClassTensor of targets")
   local targets = carry.targets:class()
   output_grad = self._module:backward({input_act, targets}, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self:inputAct()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self:inputGrad(output_grad)
   return carry
end

function SoftmaxTree:paramModule()
   return self._module
end

function SoftmaxTree:_type(type)
   self._input_type = type
   self._output_type = type
   if self._dropout then
      self._dropout:type(type)
   end
   self._module:type(type)
   return self
end

function SoftmaxTree:reset()
   self._module:reset()
   if self._sparse_init then
      self._sparseReset(self._module.weight)
   end
end

function SoftmaxTree:zeroGradParameters()
   local params, grads = self._module:parameters()
   for i=1,#grads do
      grads[i]:zero()
   end
end

-- if after feedforward, returns active parameters 
-- else returns all parameters
function SoftmaxTree:parameters()
   local params = {}
   local param_list, grad_list = self._module:parameters()
   for i=1,#param_list do
      local param = param_list[i]
      local grad = grad_list[i]
      if param:dim() == 2 then
         params['weight'..i] = { param=param, grad=grad }
      else
         params['bias'..i] = { param=param, grad=grad }
      end
   end
   return params
end

function SoftmaxTree:_zeroStatistics()
   if self._gather_stats then
      error"Not Implemented"
   end
end

function SoftmaxTree:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   for param_name, param_table in pairs(self:parameters()) do
      if param_name:find('weight') then
         if max_out_norm then
            -- rows feed into output neurons 
            dp.constrain_norms(max_out_norm, 2, param_table.param)
         end
         if max_in_norm then
            -- cols feed out from input neurons
            dp.constrain_norms(max_in_norm, 1, param_table.param)
         end
      end
   end
end

function SoftmaxTree:share(layer)
   assert(layer.isSoftmaxTree)
   layer._arrows = nil
   -- we share the hierarchy and list of nodes as is to save memory
   layer._nodes = self._nodes
   layer._parents = self._parents
   layer._children = self._children
   layer._leafs = self._leafs
   -- make sure they have the same type
   layer._module:share(self._module, 'weight', 'bias') --TODO: share grads as well
   layer._module.parentIds = self._module.parentIds
   layer._module.childIds = self._module.childIds
   layer._module.parentChildren = self._module.parentChildren
   layer._module.childParent = self._module.childParent
   layer._module.rootId = self._module.rootId
   --TODO copy other stuff over
   return self      
end

function SoftmaxTree:sharedClone()
   local clone = torch.protoClone(self, {
      input_size=self._input_size, hierarchy={[1]=torch.IntTensor{1,2,3}},
      root_id=1, sparse_init=self._sparse_init,
      dropout=self._dropout and self._dropout:clone(),
      typename=self._typename, gather_stats=self._gather_stats, 
      input_type=self._input_type, output_type=self._output_type,
      module_type=self._module_type, mvstate=self.mvstate
   })
   return self:share(clone, 'weight', 'bias')
end

function SoftmaxTree:report()
   local report = parent.report(self) or {}
   if self._gather_stats then
      error"Not Implemented"
   end
   return table.merge(report, self._report)
end
