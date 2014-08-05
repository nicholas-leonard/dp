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
      {arg='name', type='string', default='FacialKeypointFeedback',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   if baseline then
      assert(baseline:dim() == 1, "expecting 1D constant-value baseline")
      self._baseline = baseline:view(1,1,baseline:size(1))
      self._baselineSum = torch.Tensor(precision):zero()
   end
   self._precision = precision
   parent.__init(self, config)
   self._pixels = torch.range(0,precision):float():view(1,1,precision)
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._targets = torch.FloatTensor()
   self._sum = torch.Tensor(precision):zero()
   self._count = torch.Tensor(precision):zero()
   self._mse = torch.Tensor()
end

function FacialKeypointFeedback:_add(batch, output, carry, report)
   local target = batch:targets():forward('bwc')
   local act = output:forward('bwc', 'torch.FloatTensor')
   local pixels = self._pixels:expandAs(act)
   self._output:cmul(act, pixels)
   self._keypoints:sum(self._output, 3)
   self._output:cmul(target, pixels)
   self._targets:sum(self._output, 3)
   for i=1,self._keypoints:size(1) do
      local keypoint = self._keypoints[i]:select(2,1)
      local target = self._targets[i]:select(2,1)
      for j=1,self._keypoints:size(2) do
         local t = target[j]
         if t > 0.00001 then
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
   if self._n_sample > 0 then
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

