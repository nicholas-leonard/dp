------------------------------------------------------------------------
--[[ Confusion ]]--
-- Feedback
-- Adapter for optim.ConfusionMatrix
-- requires 'optim' package
------------------------------------------------------------------------
local Confusion, parent = torch.class("dp.Confusion", "dp.Feedback")
Confusion.isConfusion = true
   
function Confusion:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, name = xlua.unpack(
      {config},
      'Confusion', 
      'Adapter for optim.ConfusionMatrix',
      {arg='name', type='string', default='confusion'}
   )
   config.name = name
   parent.__init(self, config)
end

function Confusion:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Confusion:doneEpoch(report)
   if self._cm then
      print(self._id:toString().." accuracy = "..self._cm.totalValid)
   end
end

function Confusion:_add(batch, output, carry, report)
   if not self._cm then
      require 'optim'
      self._cm = optim.ConfusionMatrix(batch:targets():classes())
   end
   local act = output:forward('bf')
   if act ~= 'torch.DoubleTensor' and act ~= 'torch.FloatTensor' then
      act = output:forward('bf', 'torch.FloatTensor')
   end
   self._cm:batchAdd(act, batch:targets():forward('b'))
end

function Confusion:_reset()
   if self._cm then
      self._cm:zero()
   end
end

function Confusion:report()
   local cm = self._cm or {}
   if self._cm then
      cm:updateValids()
   end
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
      n_sample = self._n_sample
   }
end

