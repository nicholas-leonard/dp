------------------------------------------------------------------------
--[[ Model ]]--
-- Module Adapter, Component
-- Could allow for reimplementation of each nn.Module
-- to allow for automatic reshapes, set_input_space, as in pylearn2?
------------------------------------------------------------------------

local Model = torch.class("dp.Model")
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

function Model:setup(...)
   local args, mediator, id, predecessor, successor, container,
      data_view
      = xlua.unpack(
      {... or {}},
      'Model:setup', nil,
      {arg='mediator', type='dp.Mediator'},
      {arg='id', type='dp.ObjectID'},
      {arg='predecessor', type='dp.Model'},
      {arg='successor', type='dp.Model'},
      {arg='container', type='dp.CompositeModel'},
      {arg='data_view', type='string | table', 
       help='Used by a Sampler during setup to determine the view of' ..
       'input DataTensors.' ..
       "Possible values include 'image', 'imageCUDA', 'feature'"}
   )
   self._data_view = data_view
   self._mediator = mediator
   -- the id should be given by the experiment since the same 
   -- model is shared by all propagators.
   self._id = id
   -- context
   self._predecessor = predecessor
   self._successor = successor
   self._container = container
   mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Model:id()
   return self._id
end

function Model:name()
   return self._id:name()
end

function Model:tags()
   return self._tags
end

function Model:dataView()
   return self._data_view
end

--returns a report of the Model.
--if statistics were being gathered, this is the time to report them.
--Expect a report to be called at least every epoch.
function Model:report()
   
end

function Model:doneEpoch(report, ...)
   --zeros statistics
   self:zeroStatistics()
end


function Model:parameters()
   return self._params
end

function Model:setInputState(istate)
   assert(istate, "No Input State")
   if torch.isTensor(istate) then
      -- istate is tensor, then assume it is activation
      istate = {act=istate}
   end
   assert(type(istate) == 'table')
   self.istate = istate
end

function Model:setGlobalState(gstate)
   self.gstate = gstate or {}
end

function Model:setOutputState(ostate)
   if ostate == nil then
      return
   elseif torch.isTensor(ostate) then
      -- ostate is tensor, then assume it is gradients
      self.ostate.grad = ostate
      return
   end
   assert(type(ostate) == 'table')
   self.ostate = ostate
end

function Model:forward(state)
   -- state.input :
   --- input activation tensor or input state table
   -- state.global : 
   --- global state table accessible to all models in the graph
   -- state.carry :
   --- a state that is carried throughout the graph. 
   --- models may or may not use or modify it.
   --- useful when you want to forward information to a later model
   --- in the graph seperated by an unknown number of models
   self:setInputState(state.input)
   self:setGlobalState(state.global)
   local cstate = self:_forward(table.copy(state.carry)) or state.carry
   self.forwarded = true
   return self.ostate, cstate
end

function Model:_forward(cstate)
   
end

--like forward, but for evaluation purposes (valid/test).
--this is useful for stochastic Modules like Dropout, which have 
--different behavior for training than for evaluation.
function Model:evaluate(state)
   self:setInputState(state.input)
   self:setGlobalState(state.global)
   self.gstate.evaluate = true
   local cstate = self:_evaluate(table.copy(state.carry)) or state.carry
   self.evaluated = true
   self.forwarded = true
   return self.ostate, cstate
end
--default is to call forward (only diff is 'evaluate' flag in gstate)
function Model:_evaluate(cstate)
   return self:_forward(cstate)
end

function Model:backward(state)
   self:setOutputState(state.output)
   self:setGlobalState(state.global)
   local cstate = self:_backward(table.copy(state.carry)) or state.carry
   self.backwarded = true
   return self.istate, cstate
end

function Model:_backward(cstate)

end

function Model:accept(visitor)
   self.visited = true
   self:_accept(visitor)
end

function Model:_accept(visitor)
   return visitor:visitModel(self)
end

function Model:doneBatch(...)
   self:_doneBatch(...)
   if self.backwarded then
      self:zeroGradParameters()
   end
   self.forwarded = false
   self.backwarded = false
   self.evaluated = false
   self.visited = false
   self.istate = {} -- input state
   self.ostate = {} -- output state
   self.gstate = {}
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

function Model:type(type)
   -- find all tensors and convert them
   for param_name, param_table in pairs(self._params) do
      for key, tensor in pairs(param_table) do
         if torch.typename(tensor) and torch.typename(tensor):find('torch%..+Tensor') then
            param_table[key] = tensor:type(type)
         end
      end
   end
   return self
end

function Model:float()
   return self:type('torch.FloatTensor')
end

function Model:double()
   return self:type('torch.DoubleTensor')
end

function Model:cuda()
   return self:type('torch.CudaTensor')
end

function Model:reset()
end

function Model:share(mlp, ...)
end
