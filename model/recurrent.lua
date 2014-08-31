------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Is a generalization of Sequential such that it allows
-- skip and recurrent connections.
-- gets passed an additional connections table in addition to models
-- where each connection is a table with keys: source, target, and isRecurrent.
-- TODO: Subclass sequential or container?
-- TODO: Most methods are still copy pasted
-- TODO: I think most of what you wrote into container should go in here.
-- container should maybe just have the models in the topologically sorted order.
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

         --[[
         source, target,  isRecurrent = xlua.unpack(
            {conn},
            'Connections',
            'Connections between layers',
            -- TODO: Allow strings as well as dp.ObjectID as allowed types
            {arg='source', req=true,
             help='the layer the connection starts from'},
            {arg='target', req=true,
             help='the layer the connection goes to'},
            {arg='isRecurrent', type=boolean, req=false, default=false,
             help='Whether that connection should be recurrent. Defaults to' ..
                'false, unless it is from a layer to itself'})
         --]]
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
      local sortedmodels , __ , __ = g:topsort()

      models = sortedmodels
   end

   -- pass the models in topologically sorted order to container.
   config.models = models
   parent.__init(self, config)

end

function Recurrent:setup(config)
   parent.setup(self, config)
   config.container = self
   for i, model in ipairs(self._models) do
      config.id = self:id():create('s'..i)
      model:setup(config)
   end
end

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
      return self._models[1]:inputType()
   end
end

function Recurrent:outputType(output_type)
   if not output_type then
      assert(#self._models > 1, "No models to get input type")
      return self._models[#self._models]:outputType()
   end
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
