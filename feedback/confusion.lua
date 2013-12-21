require 'torch'
require 'optim'

------------------------------------------------------------------------
--[[ ConfusionMatrix ]]--
------------------------------------------------------------------------
local ConfusionMatrix = torch.class('optim.ConfusionMatrix2', 'optim.ConfusionMatrix')

function ConfusionMatrix:batchAdd(predictions, targets)
   local preds, targs, __
   if predictions:dim() == 1 then
      -- predictions is a vector of classes
      preds = predictions
   elseif predictions:dim() == 2 then
      -- prediction is a matrix of class likelihoods
      if predictions:size(2) == 1 then
         -- or prediction just needs flattening
         preds = predictions:copy()
      else
         __,preds = predictions:max(2)
      end
      preds:resize(preds:size(1))
   else
      error("predictions has invalid number of dimensions")
   end
      
   if targets:dim() == 1 then
      -- targets is a vector of classes
      targs = targets
   elseif targets:dim() == 2 then
      -- targets is a matrix of one-hot rows
      if targets:size(2) == 1 then
         -- or targets just needs flattening
         targs = targets:copy()
      else
         __,targs = targets:max(2)
      end
      targs:resize(targs:size(1))
   else
      error("targets has invalid number of dimensions")
   end
   --loop over each pair of indices
   for i = 1,preds:size(1) do
      self.mat[targs[i]][preds[i]] = self.mat[targs[i]][preds[i]] + 1
   end
end

------------------------------------------------------------------------
--[[ Confusion ]]--
-- Feedback
-- Adapter for optim.ConfusionMatrix2 
------------------------------------------------------------------------
local Confusion, parent = torch.class("dp.Confusion", "dp.Feedback")
   
function Confusion:__init(config)
   config = config or {}
   local args, name
      = xlua.unpack(
      {config},
      'Confusion', nil,
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
   print(self._id:toString() .. " accuracy = " .. self._cm.averageValid)
end

function Confusion:add(batch)
   if not self._cm then
      self._cm = optim.ConfusionMatrix2(batch:classes())
   end
   self._cm:batchAdd(batch:outputs(), batch:targets())
   self._samples_seen = self._samples_seen + batch:nSample()
end

function Confusion:reset()
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
      n_sample = self._samples_seen
   }
end
