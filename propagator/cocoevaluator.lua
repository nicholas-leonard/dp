------------------------------------------------------------------------
--[[ CocoEvaluator ]]--
------------------------------------------------------------------------
local CocoEvaluator = torch.class("dp.CocoEvaluator", "dp.Evaluator")

-- During evaluation, all instances are unknown. 
-- So the input mask only masks the padding.
-- The evaluator must iterate until the stop class has been sampled.
function CocoEvaluator:propagateBatch(batch, report) 
   self._model:evaluate()
   self:forward(batch)
   self:monitor(batch, report)
   if self._callback then
      self._callback(self._model, report)
   end
   self:doneBatch(report)
end

function CocoEvaluator:forward(batch)
   local input = batch:inputs():input()
   local target = batch:targets():input()
   local bboxTarget, classTarget = unpack(target)
   assert(not self._include_target)

   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(input:size(1)):range(1,input:size(1))
   
   self._input = self._input or input.new()
   self._input:resizeAs(self._input):copy(self._input)
   
   self._classIdx = self._classIdx or torch.LongTensor()
   self._classVal = self._classVal or torch.FloatTensor()
   
   -- for the 100 top-scoring predictions
   for i=1,self.topN or 100 do
      local output = self._model:forward(self._input, self._indices) -- second arg is for unit testing
      local bboxPred, classPred = unpack(output)
      
      if torch.type(bboxPred) ~= 'torch.FloatTensor' then
         self._bboxPred = self._bboxPred or torch.FloatTensor()
         self._classPred = self._classPred or torch.FloatTensor()
         self._bboxPred:resize(bboxPred:size()):copy(bboxPred)
         self._classPred:resize(classPred:size()):copy(classPred)
         bboxPred, classPred = self._bboxPred, self._classPred
      end
      
      if i == 1 then
         self.output = self.output or {torch.FloatTensor(), torch.LongTensor()}
         self.output[1]:resize(bboxPred:size(1), classTarget:size(2), bboxPred:size(2)):zero()
         self.output[2]:resize(classPred:size(1), classTarget:size(2)):zero()
      end
      
      local idx, offset = 1, 0
      -- for each non-STOPed sample
      for j=1,self._indices:size(1) do
         local classP = classPred:select(1,j)
         local bboxP = bboxPred:select(1,j)
         
         if classP ~= 81 then
            -- keep track of the indices of non-STOPep samples
            self._indices[idx] = self._indices[idx+offset]
            
            -- (-1,-1) top left corner, (1,1) bottom right corner of image
            local x1,y1,x2,y2 = bboxP[1], bboxP[2], bboxP[3], bboxP[4]
            -- (0,0), (1,1)
            x1, y1, x2, y2 = (x1+1)/2, (y1+1)/2, (x2+1)/2, (y2+1)/2
            -- (1,1), (256,256)
            local s = input:size(3); assert(s == input:size(4))
            x1, y1, x2, y2 = x1*(s-1)+1, y1*(s-1)+1, x2*(s-1)+1, y2*(s-1)+1
            x1, y1 = math.max(1, math.min(s,x1)), math.max(1, math.min(s,y1))
            x2, y2 = math.max(1, math.min(s,x2)), math.max(1, math.min(s,y2))
            
            -- add the predicted bounding box to the input mask
            local mask = input[{self._indices[idx],4,{},{}}]
            mask:narrow(1,y1,y2-y1+1):narrow(2,x1,x2-x1+1):fill(1)
            idx = idx + 1
         else
            offset = offset + 1
         end
      end
      
      if idx-1 == 0 then
         break -- stop when all samples predict STOP class
      end
      
      self._classVal:max(self._classIdx, classPred, 2)
      self._indices:resize(idx-1)
      self.output[1]:select(2,i):indexCopy(1, self._indices, bboxPred)
      self.output[2]:select(2,i):indexCopy(1, self._indices, self._classIdx:select(2,1))
      self._input:index(input, 1, self._indices)
   end
   
end
