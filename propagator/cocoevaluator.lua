------------------------------------------------------------------------
--[[ CocoEvaluator ]]--

------------------------------------------------------------------------
local CocoEvaluator = torch.class("dp.Evaluator", "dp.Propagator")

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
   
   for i=1,100 do
      local output = self._model:forward(self._input)
      local bboxPred, classPred = unpack(output)
      
      if i == 1 then
         self.output = self.output or {bboxPred.new(), classPred.new()}
         self.output[1]:resize(bboxPred:size(1), classTarget:size(2), bboxPred:size(2)):zero()
         self.output[2]:resize(classPred:size(1), classTarget:size(2), classPred:size(2)):zero()
      end
      
      local idx, offset = 1, 0
      for j=1,self._indices:size(1) do
         local classP = classPred:select(1,j)
         local bboxP = bboxPred:select(1,j)
         
         if classP ~= 81 then
            -- keep track of the indices of non-STOPep samples
            self._indices[idx] = self._indices[idx+offset]
            idx = idx + 1
            
            -- (-1,-1) top left corner, (1,1) bottom right corner of image
            local x1,y1,x2,y2 = bboxP[1], bboxP[2], bbox[3], bbox[4]
            -- (0,0), (1,1)
            x1, y1, x2, y2 = (x1+1)/2, (y1+1)/2, (x2+1)/2, (y2+1)/2
            -- (1,1), (256,256)
            local s = input:size(3); assert(s == input:size(4))
            x1, y1, x2, y2 = x1*(s-1)+1, y1*(s-1)+1, x2*(s-1)+1, y2*(s-1)+1
            x1, y1 = math.max(1, math.min(s,x1)), math.max(1, math.min(s,y1))
            x2, y2 = math.max(1, math.min(s,x2)), math.max(1, math.min(s,y2))
            
            -- add the predicted bounding box to the input mask
            local mask = input[{self._indices[idx+offset],4,{},{}}]
            mask:narrow(1,y1,y2-y1+1):narrow(2,x1,x2-x1+1):fill(1)
         else
            offset = offset + 1
         end
      end
      
      if idx-1 == 0 then
         break -- stop when all samples predict STOP class
      end
      
      self._indices:resize(idx-1)
      self.output[1][i]:indexCopy(1, self._indices, bboxP)
      self.output[2][i]:indexCopy(1, self._indices, classP)
      self._input:index(input,1,self._indices)
   end
   
   
end
