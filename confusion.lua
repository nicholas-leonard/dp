require 'torch'
require 'optim'

------------------------------------------------------------------------
--[[ Confusion ]]--
-- Feedback
-- Adapter for optim.ConfusionMatrix2 
------------------------------------------------------------------------
local Confusion, parent = torch.class("dp.Confusion", "dp.Feedback")
   
function Confusion.__init(config)
   local args, name
      = xlua.unpack(
      {config},
      'Confusion', nil
      {arg='name', type='string', default='confusion'}
   )
   config.name = name
   parent.__init(self, confusion)
end

function Confusion:add(batch)
   if not self._cm then
      self._cm = optim.ConfusionMatrix2(batch:classes())
   end
   self._cm:batchAdd(batch:inputs(), batch:outputs())
   self._samples_seen = self._samples_seen + batch:nSample()
end

function Confusion:reset()
   if self._cm then
      self._cm:zero()
   end
end

function Confusion:report()
   local cm = self._cm
   cm:updateValids()
   --valid means accuracy
   --union means divide valid classification by sum of rows and cols
   -- (as opposed to just cols.) minus valid classificaiton 
   -- (which is included in each sum)
   return { 
      [self:name()] = {
         matrix = cm.mat,
         per_class = { 
            accuracy = cm.valids,
            union_accuracy = cm.unionvalids,
            avg = {
               accuracy = cm.averageValid,
               union_accuracy = cm.averageUnionValid
            }
         },
         accuracy = cm.totalValid,
         avg_per_class_accuracy = cm.averageValid,
         classes = cm.classes
      },
      n_sample = self._samples_seen
   }
end
