------------------------------------------------------------------------
--[[ Learn ]]--
-- Visitor
-- Updates the parameters of parameterized models using backward 
-- propagated gradients and learning rate(s)
-- Provides for model-local learning rate scales which scale the 
-- global learning rate. 
------------------------------------------------------------------------
local Learn, parent = torch.class("dp.Learn", "dp.Visitor")
Learn.isLearn = true

function Learn:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, learning_rate, name = xlua.unpack(
      {config},
      'Learn', nil,
      {arg='learning_rate', type='number', req=true},
      {arg='name', type='string', default='learn',
       help='identifies visitor in reports.'}
   )
   self:setLearningRate(learning_rate)
   config.name = name
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.exclude = config.exclude or {}
   table.insert(config.exclude, 'no-learn')
   parent.__init(self, config)
end

function Learn:_visitModel(model)
   if not self:canVisit(model) then return end
   local params, gradParams, scales = model:parameters()
   local mvstate = model.mvstate
   for k, param in pairs(params) do
      -- learning rates can be scaled by Model or Parameter
      local learn_rate = self._learning_rate
      if mvstate.learn_scale then
         learn_rate = learn_rate * mvstate.learn_scale
      end
      if scales and scales[k] then
         -- this is useful when each param has a different scale
         learn_rate = learn_rate * scales[k]
      end
      param:add(-learn_rate, gradParams[k])
   end
end

function Learn:setLearningRate(learning_rate)
   self._learning_rate = learning_rate
end

function Learn:learningRate()
   return self._learning_rate
end
