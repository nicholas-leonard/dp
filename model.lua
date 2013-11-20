
--[[
Discussion
Statistics will be gathered for each batch. These should be 
reintialized at the start of each epoch.
Propagation phases:
experiment: propagate <-> monitor
propate: 
]]--

------------------------------------------------------------------------
--[[ Model ]]--
-- Adapter
-- Encapsulates a nn.Module such that it can be used for both 
-- optimzation and evaluation.
-- Could allow for reimplementation of each nn.Module
-- to allow for automatic reshapes, set_input_space, as in pylearn2?
------------------------------------------------------------------------

local Model = torch.class("dp.Model")

function Model:__init(module, filter)
   self._module = module
   self._report = {}
   --a table controling which statistics get gathered.
   --default is to gather statistics for parameterized modules only
   self._filter = filter
   --holds data used by ModelVisitor and this Model
   self._state = state
end

function Model:setup(...)
   local args, mediator = xlua.unpack(
      'Model:setup', nil,
      {arg='mediator', type='dp.Mediator'},
      {arg='id', type='dp.ObjectID'}
   )
   self._mediator = mediator
   self._id = id
   mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Model:id()
   return self._id
end

function Model:doneEpoch(report, ...)
   --zeros statistics
   self._report = {}
end

function Model:forward(inputs)
   --feed forward
   local outputs = self._module:forward(inputs)
   --modify state
   self._state.inputs = inputs
   self._state.outputs = outputs
   --statistics on outputs :
   
end

function Model:backward(batch)
   --statistics on gradOutputs
   
   --back propagate
   local gradInputs = self._module:backward(batch:gradOutputs())
   
end

function Model:update(batch)
   --statistics on updates
   
   --update parameters
   
   --statistics on parameters

   --update epoch statistics
end

--like forward, but for evaluation purposes (valid/test).
--this is useful for stochastic Modules like Dropout, which have 
--different behavior for training than for evaluation.
--default is to call forward (no difference)
function Model:evaluate(batch)
   return self:forward(batch)
end

--returns a report of the Model.
--if statistics were being gathered, this is the time to report them.
--Expect a report to be called every epoch.
function Model:report()
   
end

function Model:state(namespace)
   if namespace then
      return self._state[namespace]
   end
   return self._state
end

------------------------------------------------------------------------
--[[ Sequential ]]--
-- Adapter
-- Replaces nn.Sequential such that it can be used for both 
-- optimzation and evaluation.
-- TODO : reimplement nn.Sequential to work with Models instead of modules.
------------------------------------------------------------------------

local Sequential = torch.cass("dp.Sequential", "dp.Model")

function Sequential.__init()
   self._models = {}
end

function Sequential.add(model)
   table.insert(self._models, model)
end

function Sequential:report()
   
end

------------------------------------------------------------------------
--[[ Parallel ]]--
-- Adapter
-- Replaces nn.Parallel such that it can be used for both 
-- optimzation and evaluation.
------------------------------------------------------------------------
