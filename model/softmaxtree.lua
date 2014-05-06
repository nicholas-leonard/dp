------------------------------------------------------------------------
--[[ SoftmaxTree ]]--
-- A hierarchy of softmaxes.
-- Used for computing the likelihood of a leaf class.
-- Use with TreeNLL Loss.
-- Requires a tensor mapping parent_ids to child_ids. 
-- Root_id defaults to -1.
-- TODO : sum LogSoftMaxs
------------------------------------------------------------------------
local SoftmaxTree, parent = torch.class("dp.SoftmaxTree", "dp.Model")
SoftmaxTree.isSoftmaxTree = true

function SoftmaxTree:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, hierarchy, root_id, dropout, typename, 
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
   self._root_id = root_id
   self._input_size = input_size
   self._transfer = transfer
   -- index modules by parents
   parents = {}
   local children
   for parent_id, children in pairs(hierarchy) do
      assert(children:dim() == 1, "Expecting a 1D tensor of child_ids")
      local node = self.buildNode(input_size, children:size(1))
      parents[parent_id] = {node, children}
   end
   -- extract leafs and index parents by children
   leafs = {}
   children = {}
   for parent_id, node in pairs(parents) do
      local children = node[2]
      for i=1,children:size() do
         local child_id = children[i]
         if self._parents[child_id] then
            table.insert(leafs, parent_id)
         end
         children[child_id] = {parent_id, i}
      end
   end
   self._leafs = leafs
   self._children = children
   self._parents = parents
   -- temp fix until indexSelect is added to cutorch:
   self._arrows = {}
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
   local targets = carry.targets:class()
   -- When indexSelect will be part of cutorch, we can build a tree of batches.
   -- Until then, each sample has its own chain of modules which share params with a path down tree.
   local parallel = nn.ParallelTable()
   local new_targets = targets:clone()
   for i=1,activation:size(1) do
      local child_id = targets[i]
      local arrows = self._arrows[i] or {}
      self._arrows[i] = arrows
      local concat = nn.ConcatTable() --concat arrows ordered by dept
      local dept = 1
      local output_size
      while true do
         local parent_id, child_idx = unpack(self._children[child_id])
         local node, children = unpack(self._parents[parent_id])
         local arrow
         if dept == 1 then
            arrow = arrows[dept] or self.buildNode(1,1)
            output_size = node:get(1).weight:size(1)
            new_targets[i] = child_idx
         else
            arrow = arrows[dept] or self.buildArrow(1,1)
            -- only multiply probability of parent
            arrow:get(3).index = child_idx
            arrow:get(4).nfeatures = output_size
         end
         -- share params
         arrow:get(1):share(node:get(1), 'weight', 'bias')
         concat:add(arrow)
         -- cache for next batch
         arrows[dept] = arrow
         if parent_id == self._root_id then
            break
         end
         dept = dept + 1
      end
      -- sample channel
      local channel = nn.Sequential()
      channel:add(concat)
      channel:add(nn.CMulTable())
      parallel:add(channel)
   end
   self._module = nn.Sequential()
   self._module:add(nn.SplitTable(1))
   self._module:add(parallel)
   -- outputs a table of sample activation tensors
   activation = self._module:forward(activation)
   -- wrap table of tensors in dp.ComposteTensor
   self.output.act = dp.CompositeTensor{
      components = _.map(activation, function(k,v)
         return dp.DataTensor{data=v}
      end)
   }
   self._original_targets = targets
   -- so it works with NLL
   carry.targets = dp.ClassTensor{data=new_targets}
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

function SoftmaxTree:outputGrad()
   return self.output.grad:components()(self._output_type)
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

-- static method
function SoftmaxTree.buildNode(input_size, output_size)
   local node = nn.Sequential()
   node:add(nn.Linear(input_size, output_size))
   node:add(nn.SoftMax())
   return node
end

function SoftmaxTree.buildArrow(input_size, output_size)
   local node = self.buildNode(input_size, output_size)
   node:add(nn.Narrow(1, 1, 1))
   node:add(nn.Replicate(2))
   return node
end
