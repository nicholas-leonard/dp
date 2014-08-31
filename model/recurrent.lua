------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Is a generalization of Sequential such that it allows
-- skip and recurrent connections.
-- gets passed an additional connections table in addition to models
-- where each connection is a table with keys: source, target, and isRecurrent.
-- TODO: Yes, this should be a container subclass, but should sequential subclass this one?
-- TODO: Most methods are still mostly copy pasted
------------------------------------------------------------------------
local Recurrent, parent = torch.class("dp.Recurrent", "dp.Container")
Recurrent.isSequential = true

function Recurrent:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, models,  connections = xlua.unpack(
      {config},
      'Recurrent',
      'Neural network with arbitrary connections',
      {arg='models', type='table', help='a table of models'},
      {arg='connections', type='table', req=true,
       help='The the connections of the model. Table with keys: origin' ..
          'origin, and isRecurrent. The first two are the model ids.'}
   )

   config.typename = config.typename or 'recurrent'

   if connections and models then
      -- first assert all connections are valid and store them
      self._connections = {}
      --also make a temporary graph for sanity checks
      local g = graph.Graph()
      local vertices = {}
      -- fill it with the models while also asserting they are named:
      for idx, m in pairs(models) do
         vertices[m:name()] = graph.Node(m:name())
      end

      for conn_idx, conn in pairs(connections) do
         -- TODO: I would use unpack, but I don't know how to specify multiple types
         -- per model.
         local source = conn['source']
         local target = conn['target']
         local isRecurrent = conn['isRecurrent']
         -- allows specification of layer names as strings
         if (torch.type(source) == 'string' and
             torch.type(target) == 'string') then
            source = dp.ObjectID:create(source)
            target = dp.ObjectID:create(target)
         else
            assert((torch.type(source) == 'dp.ObjectID') and
                  (torch.type(target) == 'dp.ObjectID'),
               "Layer names must be strings or 'dp.ObjectID'")
         end

         if source == target then
            isRecurrent = True
         end

         local tmp_conn = {source=source, target=target, isRecurrent=isRecurrent}
         table.insert(self._connections, tmp_conn)
         -- also fill the temporary graph
         local v1 = vertices[source:name()]
         local v2 = vertices[target:name()]
         g:add(graph.Edge(v1, v2))
      end


      -- Assert the resulting structure is a DAG and keep the models in
      -- topologically sorted order --TODO: Fix in graph library for all cases
      -- TODO: More checks about the graph structure here (ie no lone nodes)
      local sortedmodels , __ , __ = g:topsort()
      -- but first map the nodes to their ids
      models = {}
      for idx, node in pairs(sortedmodels) do
         -- it is an object ID because that is what node.lua expect us to have
         table.insert(models, dp.ObjectID:create(node:label()))
      end
      -- save the graph
      self._g = g --TODO: Or not. Not needed anymore.

   end

   -- pass the models in topologically sorted order to container.
   config.models = models
   parent.__init(self, config)

   -- use the graph to initialize an appropriate view structure

   -- every graph has an input and output view, which I will save in tables for now
   -- the lookup key is the name of the model
   local incoming = {}
   local outgoing = {}

   -- TODO: As far as I can tell, the graph library has no nice way of
   -- traversing the nodes once so we are using raw node ids instead.
   -- TODO: Really do so once we get the debugging finished :P
   for __, node in pairs(models) do
      assert(torch.type(node) == 'dp.ObjectID', "Wrong node type.")
      incoming[node:name()] = {}
      outgoing[node:name()] = {}
   end

   for idx, edge in pairs(self._connections) do
      local source = edge['source']:name()
      local target = edge['target']:name()

      if not incoming[target] then
         incoming[target] = {}
      end

      if not outgoing[source] then
         outgoing[source] = {}
      end
      table.insert(incoming[target], source)
      table.insert(outgoing[source], target)
   end

   --TODO: Remove when stable and no longer needs debugging.
   self._incoming = incoming
   self._outgoing = outgoing

end

function Recurrent:setup(config)
   parent.setup(self, config)
   config.container = self
   for i, model in ipairs(self._models) do
      --TODO: Think about how to modify.
      config.id = self:id():create('r'..i)
      model:setup(config)
   end
end

-- The forward and backward gradients
function Recurrent:_forward(carry)
   local input = self.input
   for i=1,#self._models do
      if carry.evaluate then
         input, carry = self._models[i]:evaluate(input, carry)
      else
         input, carry = self._models[i]:forward(input, carry)
      end
   end
   self.output = input
   return carry
end

function Recurrent:_backward(carry)
   local output = self.output
   for i=#self._models,1,-1 do
      output, carry = self._models[i]:backward(output, carry)
   end
   self.input = output
   return carry
end

function Recurrent:inputType(input_type)
   if not input_type then
      assert(#self._models > 1, "No models to get input type")
      -- the first model in the topologically sorted order is automatically
      -- the root node
      return self._models[1]:inputType()
   end
end

function Recurrent:outputType(output_type)
   -- TODO: I think the inverse of the model graph must be some sort of DAG too
   -- as you must only have one output node of the whole thing.
   if not output_type then
      assert(#self._models > 1, "No models to get input type")
      return self._models[#self._models]:outputType()
   end
end

function Recurrent:connections()
   return self._connections
end

function Recurrent:nodes()
   return self._models
end

function Recurrent:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'dp.Recurrent'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self._models do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self._models do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self._models[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end

--[[
-- experimental
function Recurrent:flux(state)
   local output = self.output
   -- setup
   for i=1,#self._models-1 do
      self._models[i]:setSuccessor(self._models[i+1])
   end
   return self._model[1]:flux()
   self.input = output
   return carry
end
--]]
