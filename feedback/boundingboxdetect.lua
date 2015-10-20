------------------------------------------------------------------------
--[[ BoundingBoxDetect ]]--
-- Unless otherwise specified, AP and AR are averaged over multiple 
-- IoU values. Specifically we use 10 IoU thresholds of .50:.05:.95. 
-- Averaging over IoUs rewards detectors with better localization.
-- AP is averaged over all categories.
-- AP (averaged across all 10 IoU thresholds and all 80 categories) 
-- will determine the challenge winner. 
-- This should be considered the single most important metric 
-- when considering performance on COCO.
-- In COCO, there are more small objects than large objects. 
-- All metrics are computed allowing for at most 100 top-scoring
-- detections per image (across all categories).
-- Works with dp.CocoEvaluator
------------------------------------------------------------------------
local BBD, parent = torch.class("dp.BoundingBoxDetect", "dp.Feedback")

function BBD:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, n_class, name = xlua.unpack(
      {config},
      'Bounding box detection feedback', 
      'measures Precision averages over categories and IoUs',
      {arg='n_class', type='number', default=80,
       help='number of classes'},
      {arg='name', type='string', default='bbd',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   self._n_class = n_class
   self.precisionMatrix = torch.FloatTensor(self._n_class, 10):zero()
   self.classCount = torch.FloatTensor(self._n_class):zero()
   parent.__init(self, config)
end

function BBD:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function BBD:doneEpoch(report)
   self.precisionMatrix:cdiv(self.classCount:view(self._n_class, 1):expand(self._n_class, 10))
   self.precisionMatrix:apply(function(v) return _.isNaN(v) and 0 or v end)
   if self._verbose then
      print(self._id:toString().." AvgPrec = "..self.precisionMatrix:mean())
   end
end

function BBD:_add(batch, output, report)
   local bboxPred, classPred = unpack(output)
   local bboxTarget, classTarget = unpack(batch:targets():input())
   
   assert(torch.type(bboxPred) == 'torch.FloatTensor')
   assert(torch.type(classPred) == 'torch.LongTensor')
   
   if torch.type(bboxTarget) ~= 'torch.FloatTensor' then
      self._bboxTarget = self._bboxTarget or torch.FloatTensor()
      self._bboxTarget:resize(bboxTarget:size()):copy(bboxTarget)
      bboxTarget = self._bboxTarget
   end
   
   assert(torch.type(bboxTarget) == 'torch.FloatTensor')
   assert(torch.type(classTarget) == 'torch.LongTensor' or torch.type(classTarget) == 'torch.IntTensor')
   
   for i=1,bboxPred:size(1) do
      self:averagePrecision(self.precisionMatrix, bboxPred[i], classPred[i], bboxTarget[i], classTarget[i])
   end
end

-- http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
-- intersection over union
function BBD.measureIoU(bbox1, bbox2)
   local x11, y11, x12, y12 = bbox1[1], bbox1[2], bbox1[3], bbox1[4]
   local x21, y21, x22, y22 = bbox2[1], bbox2[2], bbox2[3], bbox2[4]
   -- intersection
   local x_overlap = math.max(0, math.min(x12,x22) - math.max(x11,x21));
   local y_overlap = math.max(0, math.min(y12,y22) - math.max(y11,y21));
   local overlapArea = x_overlap * y_overlap
   -- union
   local area1 = (x12 - x11) * (y12 - y11)
   local area2 = (x22 - x21) * (y22 - y21)
   local unionArea = math.max(0.000001, area1 + area2 - overlapArea)
   -- intersect over union
   local iou = overlapArea / unionArea
   assert(iou <= 1 and iou >= 0)
   return iou
end

function BBD:averagePrecision(precisionMatrix, bboxPred, classPred, bboxTarget, classTarget)
   assert(bboxPred:dim() == 2)
   assert(classPred:dim() == 1)
   assert(bboxTarget:dim() == 2)
   assert(classTarget:dim() == 1)
   assert(classTarget:max() <= self._n_class)
   
   self._grpValP = self._grpValP or classPred.new()
   self._grpIdxP = self._grpIdxP or torch.LongTensor()
   local grpP = torch.group(self._grpValP, self._grpIdxP, classPred)
   grpP[0] = nil -- remove zero-terminates
   
   self._grpValT = self._grpValT or classTarget.new()
   self._grpIdxT = self._grpIdxT or torch.LongTensor()
   local grpT = torch.group(self._grpValT, self._grpIdxT, classTarget)
   grpT[0] = nil -- remove zero-terminates
   
   self._iouThresholds = self._iouThresholds or torch.range(0.5,0.951,0.05):float()
   
   -- for each target class
   for classIdx, tblT in pairs(grpT) do
      local tblP = grpP[classIdx]
      
      if tblP then
         local classIoU = {}
         local predIds = {}
         for i=1,tblP.idx:size(1) do
            table.insert(predIds, i)
         end
         
         -- for each target instance of that class
         for i=1,tblT.val:size(1) do
            local bboxT = bboxTarget:select(1,tblT.idx[i])
            local maxIoU, maxIdx = 0, -1
            
            -- for each remaining pred instance of that class
            for j=1,#predIds do
               -- find the pred instance (of same class) with the most IoU
               local bboxP = bboxPred:select(1,predIds[j])
               local iou = self.measureIoU(bboxP, bboxT)
               
               if iou > maxIoU then
                  maxIoU = iou
                  maxIdx = j
               end
            end
            
            -- add IoU to table, and delete prediction (so it can't be used twice)
            if maxIoU >= 0.5 then
               table.insert(classIoU, maxIoU)
               table.remove(predIds,  maxIdx)
            end
         end
         
         local pm = precisionMatrix[classIdx]
         
         for i=1,10 do
            local threshold = self._iouThresholds[i]
            
            local tp = 0
            for j,iou in ipairs(classIoU) do
               if iou >= threshold then
                  tp = tp + 1
               end
            end
            
            -- true positive / (true positive + false positive)
            local precision = tp/tblP.idx:size(1)
            pm:narrow(1,i,1):add(precision)
         end
      end
      
      self.classCount:narrow(1,classIdx,1):add(1)
   end
   
   return precisionMatrix
end

function BBD:_reset()
   self.precisionMatrix:zero()
   self.classCount:zero()
end

function BBD:report()
   return { 
      [self:name()] = {
         avgprec = self.precisionMatrix:mean()
      },
      n_sample = self._n_sample
   }
end

