------------------------------------------------------------------------
--[[ FacialKeypointFeedback ]]--
-- Feedback
-- Measures error with respect to targets and optionaly compares this to 
-- constant (mean) value baseline
------------------------------------------------------------------------
local FacialKeypointFeedback, parent = torch.class("dp.FacialKeypointFeedback", "dp.Feedback")
FacialKeypointFeedback.isFacialKeypointFeedback = true

function FacialKeypointFeedback:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, precision, baseline, name = xlua.unpack(
      {config},
      'FacialKeypointFeedback', 
      'Uses mean square error to measure error w.r.t targets.'..
      'Optionaly compares this to constant (mean) value baseline',
      {arg='precision', type='number', req=true,
       help='precision (an integer) of the keypoint coordinates'},
      {arg='baseline', type='torch.Tensor', default=false,
       help='Constant baseline used for comparison'},
      {arg='name', type='string', default='facialkeypoint',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   if baseline then
      assert(baseline:dim() == 1, "expecting 1D constant-value baseline")
      self._baseline = baseline
      self._baselineSum = torch.Tensor():zero()
   end
   self._precision = precision
   parent.__init(self, config)
   self._pixels = torch.range(0,precision-1):float():view(1,1,precision)
   -- buffers
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._targets = torch.FloatTensor()
   self._max = torch.FloatTensor()
   self._indice = torch.LongTensor()
   self._sum = torch.Tensor():zero()
   self._count = torch.Tensor():zero()
   self._mse = torch.Tensor()
end

function FacialKeypointFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function FacialKeypointFeedback:_add(batch, output, report)
   -- batchSize x (nKeypoint*2) x precision
   local target = batch:targets():forward('bwc')
   assert(output:dim() == 3)
   local act = output
   if torch.type(output) ~= torch.FloatTensor() then
      self._act = self._act or torch.FloatTensor()
      self._act:resize(output:size()):copy(output)
      act = self._act
   end
   if not self._isSetup then
      self._sum:resize(act:size(2)):zero()
      self._count:resize(act:size(2)):zero()
      if self._baseline then
         self._baselineSum:resize(act:size(2)):zero()
      end
      self._isSetup = true
   end
   
   local pixels = self._pixels:expandAs(act)
   self._output:cmul(act, pixels)
   self._keypoints:sum(self._output, 3)
   self._output:cmul(target, pixels)
   self._targets:sum(self._output, 3)
   self._max:max(self._indice, target, 3)
   
   for i=1,self._keypoints:size(1) do
      local keypoint = self._keypoints[i]:select(2,1)
      local target = self._targets[i]:select(2,1)
      local maxtarget = self._max[i]:select(2,1)
      for j=1,self._keypoints:size(2) do
         local t = target[j]
         if maxtarget[j] > 0.1 then
            local err = keypoint[j] - t
            self._sum[j] = self._sum[j] + (err*err) --sum square error
            self._count[j] = self._count[j] + 1
            if (not self._baselineMse) and self._baseline then
               local err = self._baseline[j] - t
               self._baselineSum[j] = self._baselineSum[j] + (err*err)
            end
         end
      end
   end
end

function FacialKeypointFeedback:meanSquareError()
   if (not self._baselineMse) and self._baseline then
      self._baselineMse = torch.cdiv(self._baselineSum, self._count):mean()
   end
   return self._mse:cdiv(self._sum, self._count):mean()
end

function FacialKeypointFeedback:doneEpoch(report)
   if self._n_sample > 0 and self._verbose then
      local msg = self._id:toString().." MSE = "..self:meanSquareError()
      if self._baselineMse then
         msg = msg.." vs "..self._baselineMse
      end
      print(msg)
   end
end

function FacialKeypointFeedback:_reset()
   self._sum:zero()
   self._count:zero()
end

function FacialKeypointFeedback:report()
   return { 
      [self:name()] = {
         mse = self._n_sample > 0 and self:meanSquareError() or 0
      },
      n_sample = self._n_sample
   }
end

