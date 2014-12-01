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
   local args, learning_rate, verbose, name = xlua.unpack(
      {config},
      'Learn', 
      'Updates model parameters using gradients',
      {arg='learning_rate', type='number', req=true,
       help='learning rate of parameters updates'},
      {arg='verbose', type='boolean', default=true,
       help='print messages to stdout'},
      {arg='name', type='string', default='learn',
       help='identifies visitor in reports.'}
   )
   self._verbose = verbose
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
   
   -- learning rates can be scaled by Model
   local learn_scale = model.mvstate.learn_scale
   local learn_rate = self._learning_rate
   if learn_scale then
      learn_rate = learn_rate * learn_scale
   end
   
   if model.updateParameters then
      model:updateParameters(learn_rate)
      return
   end
   
   local params, gradParams, scales = model:parameters()
   for k, param in pairs(params) do
      if scales and scales[k] then
         -- parameters each have different scales
         learn_rate = learn_rate * scales[k]
      end
      param:add(-learn_rate, gradParams[k])
   end
end

function Learn:setLearningRate(learning_rate)
   if self._verbose then
      print("Learning rate = "..learning_rate)
   end
   self._learning_rate = learning_rate
end

function Learn:learningRate()
   return self._learning_rate
end

function Learn:scaleLearningRate(scale)
   self._learning_rate = self._learning_rate * scale
end
