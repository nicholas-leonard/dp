------------------------------------------------------------------------
--[[ SoftmaxTree ]]--
-- A hierarchy of softmaxes.
-- Requires a tensor mapping parent_ids to child_ids. 
-- Root_id defaults to -1.
-- TODO : sum LogSoftMaxs
------------------------------------------------------------------------
local SoftmaxTree, parent = torch.class("dp.SoftmaxTree", "dp.Model")
SoftmaxTree.isSoftmaxTree = true

function SoftmaxTree:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, hierarchy, dropout, typename, 
         sparse_init, gather_stats = xlua.unpack(
      {config},
      'SoftmaxTree', 
      'A hierarchy of softmaxes',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='hierarchy', type='table', req=true,
       help='A table mapping parent_ids to a tensor of child_ids'},
      {arg='root_id', type='number | string', default=-1,
       help='id of the root of the tree.'}
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'},
      {arg='sparse_init', type='boolean', default=true,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='gather_stats', type='boolean', default=false,
       help='gather statistics on gradients'}
   )
   self._input_size = input_size
   self._transfer = transfer
   self._tree = {}
   self._reverse_tree = {}
   for parent_id, children in pairs(hierarchy) do
      assert(children:dim() == 1, "Expecting a 1D tensor of child_ids")
      local node = nn.Sequential()
      node:add(nn.Linear(input_size, children:size(1)))
      node:add(nn.SoftMax())
      self._tree[parent_id] = {node, children}
      self._reverse_tree[
   end
   self._dropout = dropout
   self._sparse_init = sparse_init
   self._gather_stats = gather_stats
   config.typename = typename
   parent.__init(self, config)
   self:reset()
   self._tags.hasParams = true
   self:zeroGradParameters()
   self:checkParams()
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
   while true
      
   end
   activation = self._affine:forward(activation)
   if self._uncuda then
      if self._recuda == nil then
         self._recuda = (activation:type() == 'torch.CudaTensor')
      end
      activation = activation:double()
   end
   self.mvstate.affineAct = activation
   activation = self._transfer:forward(activation)
   -- wrap torch.Tensor in a dp.DataTensor
   self.output.act = dp.DataTensor{data=activation}
   return carry
end

function SoftmaxTree:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local input_act = self.mvstate.affineAct
   local output_grad = self:outputGrad()
   output_grad = self._transfer:backward(input_act, output_grad, scale)
   if self._recuda then
      output_grad = output_grad:cuda()
   end
   self.mvstate.affineGrad = output_grad
   input_act = self.mvstate.dropoutAct or self.input.act:feature()
   output_grad = self._affine:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self.input.act:feature()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self.input.grad = self.input.act:featureClone(output_grad)
   return carry
end

function SoftmaxTree:paramModule()
   return self._affine
end

function SoftmaxTree:_type(type)
   self._input_type = type
   self._output_type = type
   self._affine:type(type)
   if type ~= 'torch.CudaTensor' and not self._uncuda then
      self._transfer:type(type)
   end
   if self._dropout then
      self._dropout:type(type)
   end
   return self
end


local function hsoftmax_test()
   local sol = nn.SparseOutLinear(100, 1000, true, true)
   sol:forward(input, outputIndices)
end
