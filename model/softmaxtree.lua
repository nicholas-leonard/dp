------------------------------------------------------------------------
--[[ SoftmaxTree ]]--
-- A hierarchy of softmaxes.
-- Used for computing the likelihood of a leaf class.
-- Use with TreeNLL Loss.
-- Requires a tensor mapping parent_ids to child_ids. 
-- Root_id defaults to -1.
-- TODO : sum LogSoftMaxs, cache modules
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
      {arg='root_id', type='number | string', default=-1,
       help='id of the root of the tree.'},
      {arg='typename', type='string', default='softmaxtree', 
       help='identifies Model type in reports.'}
   )
   self._root_id = root_id
   self._input_size = input_size
   -- index modules by parents
   parents = {}
   -- any container would do here. We just use it for changing types.
   self._nodes = nn.Parallel()
   local children
   for parent_id, children in pairs(hierarchy) do
      assert(children:dim() == 1, "Expecting a 1D tensor of child_ids")
      local node = self.buildNode(input_size, children:size(1))
      parents[parent_id] = {node, children}
      self._nodes:add(node)
   end
   -- extract leafs and index parents by children
   leafs = {}
   self._children = {}
   for parent_id, node in pairs(parents) do
      local children = node[2]
      for i=1,children:size(1) do
         local child_id = children[i]
         if parents[child_id] then
            table.insert(leafs, parent_id)
         end
         self._children[child_id] = {parent_id, i}
      end
   end
   self._leafs = leafs
   self._parents = parents
   -- temp fix until indexSelect is added to cutorch:
   self._arrows = {}
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
   -- When indexSelect will be part of cutorch, we could ostensibly
   -- build a tree of batches.
   -- Until then, each sample has its own chain of modules which share 
   -- params with a path down tree.
   local parallel = nn.Parallel(1,1)
   self._active_nodes = {}
   for i=1,activation:size(1) do
      local child_id = targets[i]
      local arrows = self._arrows[i] or {}
      self._arrows[i] = arrows
      local concat = nn.ConcatTable() --concat arrows ordered by dept
      local dept = 1
      while true do
         local parent_id, child_idx = unpack(self._children[child_id])
         local node, children = unpack(self._parents[parent_id])
         table.insert(self._active_nodes, node)
         local arrow = arrows[dept] or self.buildArrow(1,1)
         -- only multiply probability of parent
         arrow:get(3).index = child_idx
         -- share params
         local arrow_linear = arrow:get(1)
         local node_linear = node:get(1)
         arrow_linear:share(node_linear, 'weight', 'bias')
         -- resize gradients
         arrow_linear.gradWeight:resizeAs(node_linear.gradWeight):zero()
         arrow_linear.gradBias:resizeAs(node_linear.gradBias):zero()
         concat:add(arrow)
         -- cache for next batch
         arrows[dept] = arrow
         if parent_id == self._root_id then
            break
         end
         child_id = parent_id
         dept = dept + 1
      end
      -- sample channel (one channel per sample)
      local channel = nn.Sequential()
      channel:add(concat)
      channel:add(nn.CMulTable())
      parallel:add(channel)
   end
   -- make sure it has the right type
   self._module = nn.Sequential()
   self._module:add(parallel)
   self._module:add(nn.Reshape(activation:size(1), 1))
   self._module:type(self:moduleType())
   -- outputs a column vector of likelihoods of targets
   activation = self._module:forward(activation)
   self:outputAct(activation)
   return carry
end

function SoftmaxTree:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local input_act = self.mvstate.dropoutAct or self:inputAct()
   local output_grad = self:outputGrad()
   output_grad = self._module:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self:inputAct()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self:inputGrad(output_grad)
   return carry
end

-- if after feedforward, returns active nodes 
-- else returns all nodes
function SoftmaxTree:paramModule()
   if self.forwarded then
      return self._module
   end
   print"SoftmaxTree:paramModule : warning returning all modules"
   return self._nodes
end

function SoftmaxTree:_type(type)
   self._input_type = type
   self._output_type = type
   self._nodes:type(type)
   if self._dropout then
      self._dropout:type(type)
   end
   if self._module then
      self._module:type(type)
   end
   return self
end

function SoftmaxTree:reset()
   self._nodes:reset()
   if self._sparse_init then
      for i, node in ipairs(self._nodes) do
         self._sparseReset(node:get(1).weight)
      end
   end
end

-- if after feedforward, returns active parameters 
-- else returns all parameters
function SoftmaxTree:parameters()
   local params = {}
   local nodes
   if self.forwarded then
      nodes = self._active_nodes
   else
      print"SoftmaxTree:parameters : warning returning all parameters"
      nodes = _.map(self._parents, function(k,v) return v[1] end)
   end
   for i,node in ipairs(nodes) do
      local module = node:get(1)
      params['weight'..i] = { param=module.weight, grad=module.gradWeight }
      params['bias'..i] = { param=module.bias, grad=module.gradBias }
   end
   return params
end

function SoftmaxTree:_zeroStatistics()
   if self._gather_stats then
      error"Not Implemented"
      for param_name, param_table in pairs(self:parameters()) do
         self._stats[param_name] = {
            grad={sum=0, mean=0, min=0, max=0, count=0, std=0}
         }
      end
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
   layer:type(self._module_type)
   return self      
end

function SoftmaxTree:sharedClone()
   local clone = dp.SoftmaxTree{
      input_size=self._input_size, hierarchy={1,torch.IntTensor(1,2,3)},
      root_id=self._root_id, sparse_init=self._sparse_init,
      dropout=self._dropout and self._dropout:clone(),
      typename=self._typename, gather_stats=self._gather_stats, 
      input_type=self._input_type, output_type=self._output_type,
      module_type=self._module_type, mvstate=self.mvstate
   }
   return self:share(clone, 'weight', 'bias')
end

function SoftmaxTree:report()
   local report = parent.report(self) or {}
   if self._gather_stats then
      error"Not Implemented"
      for param_name, param_table in pairs(self:parameters()) do
         local param_stats = self._stats[param_name]
         if param_stats and param_stats.grad and param_stats.grad.count > 0 then
            local grad = param_stats.grad
            local param_report = self._report[param_name] or {}
            local count = grad.count
            local grad_report = {
                  sum=grad.sum/count, mean=grad.mean/count,
                  min=grad.min/count, max=grad.max/count,
                  std=grad.std/count, count=grad.count
            }
            self._report[param_name] = {grad=grad_report}
         end
      end
   end
   return table.merge(report, self._report)
end

-- static method
function SoftmaxTree.buildNode(input_size, output_size)
   local node = nn.Sequential()
   node:add(nn.Linear(input_size, output_size))
   node:add(nn.SoftMax())
   return node
end

function SoftmaxTree.buildArrow(input_size, output_size)
   local node = SoftmaxTree.buildNode(input_size, output_size)
   node:add(nn.Narrow(1, 1, 1))
   return node
end
