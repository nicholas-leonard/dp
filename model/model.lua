--[[ TODO ]]--
-- to allow for automatic reshapes, set_input_space, as in pylearn2?
-- remove predecessor, successor, etc?
-- remove self._params

------------------------------------------------------------------------
--[[ Model ]]--
-- Adapter of nn.Modules
------------------------------------------------------------------------
local Model, parent = torch.class("dp.Model", "dp.Node")
Model.isModel = true

function Model:__init(...)
   local args, typename, tags, mvstate = xlua.unpack(
      {... or {}},
      'Model', nil,
      {arg='typename', type='string', req=true, 
       help='identifies Model type in reports.'},
      {arg='tags', type='table', default={},
       help='table of tags used for determining which visitors ' ..
       'are allowed to visit the model'},
      {arg='mvstate', type='table', default={},
       help='model-visitor state'}
   )
   self._typename = typename
   self._params = {} -- parameters
   self._stats = {} -- statistics
   self._tags = tags -- tags
   self._report = {}
   self.mvstate = mvstate
   self:doneBatch()
end

function Model:setup(config)
   local args, predecessor, successor, container
      = xlua.unpack(
      {config or {}},
      'Model:setup', nil,
      {arg='predecessor', type='dp.Model'},
      {arg='successor', type='dp.Model'},
      {arg='container', type='dp.Container'}
   )
   -- context
   self._predecessor = predecessor
   self._successor = successor
   self._container = container
   parent.setup(self, config)
end

function Model:tags()
   return self._tags
end

function Model:doneEpoch(report, ...)
   --zeros statistics between epochs
   self:zeroStatistics()
end

function Model:parameters()
   return self._params
end

function Model:accept(visitor)
   self.visited = true
   self:_accept(visitor)
end

function Model:_accept(visitor)
   return visitor:visitModel(self)
end

function Model:doneBatch(...)
   parent.doneBatch(self, ...)
   if self.backwarded then
      self:zeroGradParameters()
   end
   self.visited = false
   self.ostate = {} -- output state
end

function Model:_doneBatch(...)
end

function Model:zeroGradParameters()
   for param_name, param_table in pairs(self:parameters()) do
      param_table.grad:zero()
   end
end

function Model:zeroStatistics()
end

function Model:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function Model:share(mlp, ...)
   error"Not Implemented"
end
