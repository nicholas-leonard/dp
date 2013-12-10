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
Learn.isLearn = true

function Learn:__init(config)
   local args, learning_rate, name = xlua.unpack(
      {config},
      'Learn', nil,
      {arg='learning_rate', type='number', req=true},
      {arg='name', type='string', default='learn',
       help='identifies visitor in reports.'}
   )
   self:setLearningRate(learning_rate)
   config.include = config.include or {}
   config.name = name
   table.insert(config.include, 'hasParams')
   parent.__init(self, config)
end

function Learn:_visitModel(model)
   if not self:canVisit(model) then return end
   local params = model:parameters()
   for param_name, param_table in pairs(params) do
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
               -self._learning_rate, param_table.delta
            )
      else
         param_table.param:add(
               -self._learning_rate, param_table.grad
            )
      end
   end
end

function Learn:setLearningRate(learning_rate)
   self._learning_rate = learning_rate
end

function Learn:learningRate()
   return self._learning_rate
end
