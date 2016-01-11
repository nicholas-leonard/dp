------------------------------------------------------------------------
--[[ Confusion ]]--
-- Feedback
-- Adapter for optim.ConfusionMatrix
-- requires 'optim' package
------------------------------------------------------------------------
local Confusion, parent = torch.class("dp.Confusion", "dp.Feedback")
Confusion.isConfusion = true
   
function Confusion:__init(config)
   require 'optim'
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, bce, name, target_dim, output_module = xlua.unpack(
      {config},
      'Confusion', 
      'Adapter for optim.ConfusionMatrix',
      {arg='bce', type='boolean', default=false,
       help='set true when using Binary Cross-Entropy (BCE)Criterion'},
      {arg='name', type='string', default='confusion',
       help='name identifying Feedback in reports'},
      {arg='target_dim', default=-1, type='number',
      help='row index of target label to be used to measure confusion'},
      {arg='output_module', type='nn.Module',
       help='module applied to output before measuring confusion matrix'}
   )
   config.name = name
   self._bce = bce
   self._output_module = output_module or nn.Identity()
   self._target_dim = target_dim
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

function Confusion:_add(batch, output, report)
   if self._output_module then
      output = self._output_module:updateOutput(output)
   end
   
   if not self._cm then
      if self._bce then
         self._cm = optim.ConfusionMatrix({0,1})
      else
         self._cm = optim.ConfusionMatrix(batch:targets():classes())
      end
      self._cm:zero()
   end
   
   local act = self._bce and output:view(-1) or output:view(output:size(1), -1)
   local tgt = batch:targets():forward('b')
   if self._target_dim >0 then
      tgt=tgt[self._target_dim]
   end
   
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
   
   if not (torch.isTypeOf(act,'torch.FloatTensor') or torch.isTypeOf(act, 'torch.DoubleTensor')) then
      self._actf = self.actf or torch.FloatTensor()
      self._actf:resize(act:size()):copy(act)
      act = self._actf
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

