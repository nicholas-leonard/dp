------------------------------------------------------------------------
--[[ Visitor ]]--
-- Visits a composite struture of Models and modifies their states.

-- TODO: 
-- Visitors should try to access a model method assigned to 
-- each visitor (if exists). This would allow models to implement
-- visitor specifics. (already started with dp.Linear model)
-- Visitors accumulate statistics for reporting purposes
-- Visitor statistics
------------------------------------------------------------------------
local Visitor = torch.class("dp.Visitor")

function Visitor:__init(...)
   local args, name, include, exclude = xlua.unpack(
      {... or {}},
      'Visitor', nil,
      {arg='name', type='string', req=true,
       help='identifies visitor in reports.'},
      {arg='include', type='table',
       help='only models having a true value for the member named ' .. 
       'in this table are visited, unless the member is also listed ' ..
       'in the exclude table, in this case it is not visited. ' ..
       'If include is empty, all models are included, unless ' ..
       'specified in the exclude list'},
      {arg='exclude', type='table', default={},
       help='models having a member named in this table are not ' ..
       'visited, even if the member is in the include table, i.e. ' ..
       'exclude has priority over include'}
   )
   self._name = name
   self._exclude = exclude
   self._include = include
end

function Visitor:id()
   return self:_id
end

function Visitor:name()
   return self._id:name()
end

-- compares model to filter to see if it can be visited
function Visitor:canVisit(model)
   local model_tags = model:tags()
   if not self._exclude or table.eq(self._exclude, {}) then
      for tag in ipairs(self._exclude) do
         if model_tags[tag] then
            return false
         end
      end
   end
   if not self._include or table.eq(self._include, {}) then
      for tag in ipairs(self._include) do
         if model_tags[tag] then
            return true
         end
      end
   else
      return true
   end
   return false
end

function Visitor:visitModel(model)
   -- can we visit model?
   if not self:canVisit(model) then 
      return 
   end
   --TODO : mvstate[self:id():parent():name()][self:name()]
   -- or mvstate[self._id_string] where self._id_string = self._id:toString())
   -- has the model-visitor state been initialized?
   if not model.mvstate[self:id():name()] then 
      model.mvstate[self:id():name()] = {}
   end
   self._visitModel(model)
end

function Visitor:_visitModel(model)
   return
end

--default is to do nothing for visitors (for now)
function Visitor:visitContainer(model)
   
end

function Visitor:report()
   return {[self:name()] = {}}
end

function Visitor:setup(...)
   local args, mediator = xlua.unpack(
      {... or {}},
      'Visitor:setup', nil,
      {arg='mediator', type='dp.Mediator'},
      {arg='model', type='dp.Model'},
      {arg='propagator', type='dp.Propagator'}
   )
   self._mediator = mediator
   -- not sure including model is good idea...
   self._model = model
   self._propagator = propagator
   self._id = propagator:id():create(self._name)
   self._name = nil
end


------------------------------------------------------------------------
--[[ WeightDecay ]]--
-- ModelVisitor
-- Decays the weight of the visited parameterized models.
------------------------------------------------------------------------

local WeightDecay, parent = torch.class("dp.WeightDecay", "Visitor")

function WeightDecay:__init(config)
   config = config or {}
   local args, wd_factor, name = xlua.unpack(
      {config},
      'WeightDecay', nil,
      {arg='wd_factor', type='number', help='Weight decay factor'},
      {arg='name', type='string', default='weightdecay',
       help='identifies visitor in reports.'}
   )
   self._wd_factor = wd_factor
   config.include = config.include or {}
   config.name = name
   table.insert(config.include, 'hasParams')
   parent.__init(self, config)
end

function WeightDecay:_visitModel(model)
   local params = model:parameters()
   for param_name, param_table in pairs(params) do
      if param_name ~= 'bias' then
         param_table.grad:add(self.wd_factor, param_table.param)
      end
   end
end


------------------------------------------------------------------------
--[[ Momentum ]]--
-- ModelVisitor
-- Applies momentum to parameters
------------------------------------------------------------------------

local Momentum, parent = torch.class("dp.Momentum", "Visitor")

function Momentum:__init(config)
   config = config or {}
   local args, momentum_factor, dampling_vactor, nesterov, name
      = xlua.unpack(
      {config},
      'Momentum', 'Applies momentum to parameters',
      {arg='momentum_factor', type='number', req=true},
      {arg='damping_factor', type='number', default=0},
      {arg='nesterov', type='boolean', default=false},
      {arg='name', type='string', default='momentum',
       help='identifies visitor in reports.'}
   )
   --Damping is an influence within or upon an oscillatory system that 
   --has the effect of reducing, restricting or preventing its 
   --oscillations. In physical systems, damping is produced by 
   --processes that dissipate the energy stored in the oscillation
   self._momentum_factor = momentum_factor
   self._damping_factor = damping_factor
   self._nesterov = nesterov
   config.include = config.include or {}
   config.name = name
   table.insert(config.include, 'hasParams')
   parent.__init(self, config)
end

function Momentum:_visitModel(model)
   local params = model:parameters()
   
   for param_name, param_table in pairs(params) do
      if not param_table.past_grad then
         param_table.past_grad 
            = torch.Tensor():typeAs(
                  param_table.grad
               ):resizeAs(
                  param_table.grad
               ):copy(
                  param_table.grad
               )
      else
         param_table.past_grad:mul(
               self._momentum_factor
            ):add(
               1-self._damping_factor, 
               param_table.grad
            )
      end
      if self._nesterov then
         param_table.grad:add(
               self._momentum_factor, 
               param_table.past_grad
            )
      else
         param_table.grad = param_table.past_grad
      end
   end
end


------------------------------------------------------------------------
--[[ MaxNorm ]]--
-- ModelVisitor
-- Hard constraint on the upper bound of the norm of output and/or input
-- neuron weights (in a weight matrix). Has a regularization effect 
-- analogous to WeightDecay, but with easier to optimize 
-- hyper-parameters. Quite useful with Rectified Linear Units (ReLU).
-- Should occur after LearningRate in VisitorChain
------------------------------------------------------------------------

local MaxNorm, parent = torch.class("dp.MaxNorm", "Visitor")

function MaxNorm:__init(config)
   config = config or {}
   local args, max_col_norm, max_row_norm, name = xlua.unpack(
      {config},
      'MaxNorm', 
      'Hard constraint on the upper bound of the norm of output ' ..
      'and input weights.',
      {arg='max_out_norm', type='number', default=1
       help='max norm of output neuron weights'},
      {arg='max_in_norm', type='number', 
      help='max norm of input neuron weights'},
      {arg='name', type='string', default='maxnorm',
       help='identifies visitor in reports.'}
   )
   self._max_out_norm = max_out_norm
   self._max_in_norm = max_in_norm
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.name = name
   parent.__init(self, config)
end

function MaxNorm:_visitModel(model)
   if model.maxNorm then
      model:maxNorm(self._max_out_norm, self._max_in_norm)
      return
   else
      print"Warning: MaxNorm not implemented for model " .. model ..
      ". Ignoring model-visitor pair"
   end
   
   --[[
   local params = {}
   local model_params = model:parameters()
   local bias = model_params.bias
   if not bias then
      if not model.mvstate[self:id():name()].warned then
         print"MaxNorm Warning: model " .. model .. 
         " has no bias. " .. "Assuming columns of the weight " ..
         "matrix are outgoing weights."
      end
      model.mvstate[self:id():name()].warned = true
   else
      bias
   end
   for param_name, param_table in pairs(model_params) do
      
   end
   ]]--
end


------------------------------------------------------------------------
--[[ Learn ]]--
-- Visitor
-- Updates the parameters of parameterized models using backward 
-- propagated gradients and learning rate(s)
-- Provides for model-local learning rate scales which scale the 
-- global learning rate. 

-- TODO : 
-- Provide interface for mstate (model state) construction.
--  This would allow for initialization of visitor-model variables like
--  learning scalers, etc.
------------------------------------------------------------------------
local Learn, parent = torch.class("dp.Learn", "dp.Visitor")
LearningRate.isLearningRate = true

function Learn:__init(config)
   local args, learning_rate, name = xlua.unpack(
      {config},
      'Learn', nil,
      {arg='learning_rate', type='number', req=true},
      {arg='name', type='string', default='learn',
       help='identifies visitor in reports.'}
   )
   self._learning_rate = learning_rate
   config.include = config.include or {}
   config.name = name
   table.insert(config.include, 'hasParams')
   parent.__init(self, config)
end

function Learn:visitModel(model)
   if not self:canVisit(model) then return end
   
   local params = model:parameters()
   for param_name, param_table in pairs(param) do
      -- parameter update with single or individual learning rates
      if param_table.learning_rate_scale then
         if not param_table.delta then
            param_table.delta 
               = torch.Tensor():typeAs(
                     param_table.param
                  ):resizeAs(
                     param_table.grad
                  )
         end
         param_table.delta:copy(
               param_table.learning_rate_scale
            ):cmul(
               param_table.grad
            )
         param_table.param:add(
               -self._learning_rate, 
               param_table.delta
            )
      else
         param_table.param:add(
               -self._learning_rate, 
               param_table.grad
            )
      end
   end
end

------------------------------------------------------------------------
--[[ VisitorChain ]]--
-- Composite, Visitor, Chain of Responsibility
------------------------------------------------------------------------
local VisitorChain, parent = torch.class("dp.VisitorChain", "dp.Visitor")

function VisiorChain:__init(config)
   local args, visitors = xlua.unpack(
      {config}
      'VisitorChain', nil,
      {arg='visitors', type='table', req=true}
   )
   config.name = 'visitorchain'
   parent.__init(self, config)
   self._visitors = visitors
end


function VisitorChain:setup(config)
   parent.setup(self, config)
   for i, visitor in ipairs(self._visitors) do
      visitor:setup(config)
   end
end

function VisitorChain:_visitModel(model)
   for i, visitor in ipairs(self._visitors) do
      visitor:visitModel(model)
   end
end


function VisitorChain:report()
   -- merge reports
   local report = {}
   for k, visitor in pairs(self._visitors) do
      merge(report, visitor:report())
   end
   return report
end
