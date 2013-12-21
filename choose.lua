------------------------------------------------------------------------
--[[ Choose ]] --
-- Uniformly Chooses one of the provided options.
------------------------------------------------------------------------  
local Choose = torch.class("dp.Choose")
Choose.isChoose = true

function Choose:__init(options)
   self._options = options
end

function Choose:sample()
   return self._options[math.random(#self._options)]
end

function Choose:report()
   return {typename = torch.typename(self), options = self._options}
end

------------------------------------------------------------------------
--[[ WeightedChoose ]] --
-- Choose options by sampling from a multinomial probability distribution
-- proportional to their respective weight.
------------------------------------------------------------------------  
local WeightedChoose, parent = torch.class("dp.WeightedChoose", "dp.Choose")
WeightedChoose.isWeightedChoose = true

-- Distribution is a table of where keys are options, 
-- and their values are weights.
function WeightedChoose:__init(distribution)
   assert(type(distribution) == 'table')
   self._size = _.size(distribution)
   local probs = {}
   local options = {}
   for k,v in pairs(distribution) do
      table.insert(probs, v)
      table.insert(options, k)
   end
   self._probs = torch.DoubleTensor(probs)
   parent.__init(self, options)
end

function WeightedChoose:sample()
   local index = dp.multinomial(self._probs)[1]
   return self._options[index]
end

function WeightedChoose:report()
   local report = parent.report(self)
   report.probs = self._probs
   return report
end

------------------------------------------------------------------------
--[[ TimeChoose ]]--
-- each sample is the time from os.time().
-- useful for setting the random seed of experiments
------------------------------------------------------------------------
local TimeChoose = torch.class("dp.TimeChoose")
TimeChoose.isChoose = true
TimeChoose.isTimeChoose = true

function TimeChoose:sample()
   return os.time()
end

function TimeChoose:report()
   return {typename = torch.typename(self)}
end
