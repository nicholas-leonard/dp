------------------------------------------------------------------------
--[[ Sequential ]]--
-- Model, Adapter, Composite
-- Replaces nn.Sequential such that it can be used for both 
-- optimzation and evaluation.
------------------------------------------------------------------------
local Sequential, parent = torch.class("dp.Sequential", "dp.Container")
Sequential.isSequential = true

function Sequential:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   config.typename = config.typename or 'sequential'
   parent.__init(self, config)
end

function Sequential:setup(config)
   parent.setup(self, config)
   config.container = self
   for i, model in ipairs(self._models) do
      config.id = self:id():create('s'..i)
      model:setup(config)
   end
end

function Sequential:_forward(carry)
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

function Sequential:_backward(carry)
   local output = self.output
   for i=#self._models,1,-1 do
      output, carry = self._models[i]:backward(output, carry)
   end
   self.input = output
   return carry
end

function Sequential:inputType(input_type)
   if not input_type then
      assert(#self._models > 1, "No models to get input type") 
      return self._models[1]:inputType()
   end
end

function Sequential:outputType(output_type)
   if not output_type then
      assert(#self._models > 1, "No models to get input type") 
      return self._models[#self._models]:outputType()
   end
end

function Sequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'dp.Sequential'
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
function Sequential:flux(state)
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
