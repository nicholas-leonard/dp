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
   local args, bce, name = xlua.unpack(
      {config},
      'Confusion', 
      'Adapter for optim.ConfusionMatrix',
      {arg='bce', type='boolean', default=false,
       help='set true when using Binary Cross-Entropy (BCE)Criterion'},
      {arg='name', type='string', default='confusion',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   self._bce = bce
   parent.__init(self, config)
end

function Confusion:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Confusion:doneEpoch(report)
   if self._cm and self._verbose then
      print(self._id:toString().." accuracy = "..self._cm.totalValid)
   end
end

function Confusion:_add(batch, output, carry, report)
   if not self._cm then
      require 'optim'
      self._cm = optim.ConfusionMatrix(batch:targets():classes())
   end
   
   -- binary cross-entropy has one output
   local view = self._bce and 'b' or 'bf'
   local act = output:forward(view)
   local act_type = torch.type(act)
   if act_type ~= 'torch.DoubleTensor' and act_type ~= 'torch.FloatTensor' then
      act = output:forward(view, 'torch.FloatTensor')
   end
   
   local tgt = batch:targets():forward('b')
   
   if self._bce then
      self._act = self._act or act.new()
      self._tgt = self._tgt or tgt.new()
      -- round it to get a class
      -- add 1 to get indices starting at 1
      self._act:gt(act, 0.5):add(1) 
      self._tgt:add(tgt,1)
      act = self._act
      tgt = self._tgt
   end

   self._cm:batchAdd(act, tgt)
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

