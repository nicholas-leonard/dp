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
   assert(config.connections, "Please specify the connections of the model")
   config.typename = config.typename or 'recurrent'

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
