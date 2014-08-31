------------------------------------------------------------------------
--[[ dp.Container ]]--
-- Composite of Model Components
------------------------------------------------------------------------
local Container, parent = torch.class("dp.Container", "dp.Model")
require 'graph'
Container.isContainer = true

function Container:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   config = config or {}
   local args, models, connections = xlua.unpack(
      {config},
      'Container',
      'Composite of Model Components',
      {arg='models', type='table', help='a table of models'},
      {arg='connections', type='table', req=false,
       help='If the container is not just a sequential mlp, specify each connection'}
   )
   self._models = {}
   parent.__init(self, config)

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

      xlua.print(sortedmodels)
      models = sortedmodels

   elseif models then
      self:extend(models)
   else --empty container
   end


end

function Container:extend(models)
   for model_idx, model in pairs(models) do
      self:add(model)
   end
end

function Container:add(model)
   assert(not self._setup,
      "Should insert models before calling setup()!")
   table.insert(self._models, model)
end

function Container:size()
   return #self._models
end

function Container:get(index)
   return self._models[index]
end

function Container:inputType(input_type)
   error"Not Implemented"
end

function Container:outputType(output_type)
   error"Not Implemented"
end

function Container:_type(type)
   -- find submodels in classic containers 'models'
   if not _.isEmpty(self._models) then
      for i, model in ipairs(self._models) do
         model:type(type)
      end
   end
end

function Container:_accept(visitor)
   for i=1,#self._models do
      self._models[i]:accept(visitor)
   end
   visitor:visitContainer(self)
end

function Container:report()
   -- merge reports
   local report = {typename=self._typename}
   for k, model in ipairs(self._models) do
      report[model:name()] = model:report()
   end
   return report
end

function Container:doneBatch(...)
   for i=1,#self._models do
      self._models[i]:doneBatch(...)
   end
   -- stops parent from calling zeroGradParameters (again)
   self.backwarded = false
   parent.doneBatch(self, ...)
end

function Container:zeroGradParameters()
   for i=1,#self._models do
      self._models[i]:zeroGradParameters()
   end
end

function Container:reset(stdv)
   for i=1,#self._models do
      self._models[i]:reset(stdv)
   end
end

function Container:parameters()
   error"NotImplementedError"
end
