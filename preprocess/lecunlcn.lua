-----------------------------------------------------------------------
--[[ LeCunLCN ]]--
-- Performs Local Contrast Normalization on images
-- http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
-----------------------------------------------------------------------
local LeCunLCN = torch.class("dp.LeCunLCN", "dp.Preprocess")
LeCunLCN.isLeCunLCN = true

function LeCunLCN:__init(config)
   local args
   args, self._kernel_size, self._threshold, self._batch_size, self._channels,
      self._progress = xlua.unpack(
      {config},
      'LeCunLCN', 
      'LeCunLCN constructor',
      {arg='kernel_size', type='number', default=9, 
       help='local contrast kernel size'},
      {arg='threshold', type='number', default=1e-4,
       help='threshold for denominator'},
      {arg='batch_size', type='number', default=256,
       help='batch_size used for performing the preprocessing'},
      {arg='channels', type='table',
       help='List of channels to normalize. Defaults to {1,2,3}'},
      {arg='progress', type='boolean', default=true, 
       help='display progress bar'}
   )
   self._sampler = dp.Sampler{batch_size = batch_size}
   self._channels = self._channels or {1,2,3}
   self._filter = self.gaussianFilter(self._kernel_size)
   -- buffers
   self._convout = torch.Tensor()
   self._center = torch.Tensor()
   self._square = torch.Tensor()
   self._mean = torch.Tensor()
   self._divisor = torch.Tensor()
   self._result = torch.Tensor()
   self._denom = torch.Tensor()
end

-- static method
function LeCunLCN.gaussianFilter(kernel_size)
   local x = torch.zeros(kernel_size, kernel_size)
   local _gauss = function(x, y, sigma)
      sigma = sigma or 2.0
      local Z = 2 * math.pi * math.pow(sigma, 2)
      return 1 / Z * math.exp(-(math.pow(x,2)+math.pow(y,2))/(2 * math.pow(sigma,2)))
   end
   
   local mid = math.ceil(kernel_size / 2)
   for i = 1, kernel_size do
      for j = 1, kernel_size do
         x[i][j] = _gauss(i-mid, j-mid)
      end
   end
   
   return x / x:sum()
end

function LeCunLCN:apply(dv, can_fit)   
   for stop, view in dv:ipairsSub(self._batch_size, true, true) do
      -- transform and replace original tensor
      view:replace("bhwc", 
         self:transform(
            view:forward("bhwc")
         ), true
      )
      
      if self._progress then
         -- display progress
         xlua.progress(stop, dv:nSample())
      end
   end
   
   -- for aesthetics :
   if self._progress then
      xlua.progress(dv:nSample(), dv:nSample())
   end
end

-- expects x to have view 'bhwc'
function LeCunLCN:transform(x)
   if torch.type(x) ~= torch.type(self._filter) then
      self._filter = self._filter:type(torch.type(x))
      self._convout = self._convout:type(torch.type(x))
      self._center = self._center:type(torch.type(x))
      self._square = self._square:type(torch.type(x))
      self._mean = self._mean:type(torch.type(x))
      self._divisor = self._divisor:type(torch.type(x))
      self._denom = self._denom:type(torch.type(x))
      self._result = self._result:type(torch.type(x))
   end
   
   self._result:resizeAs(x):copy(x)
   for i,channelIdx in ipairs(self._channels) do
      assert(torch.type(channelIdx) == 'number') 
      assert(channelIdx >= 0 and channelIdx <= x:size(4))

      self._result:select(4,channelIdx):copy(self:normalize(x:select(4,channelIdx)))
   end
   return self._result
end

-- expects input to have view 'bhw'
function LeCunLCN:normalize(input)   
   local filter, convout = self._filter, self._convout
   local center, square = self._center, self._square
   local mean, divisor, denom = self._mean, self._divisor, self._denom
   filter = filter:view(1, filter:size(1), filter:size(2))
   filter = filter:expand(input:size(1), filter:size(2), filter:size(3))
   convout:conv2(input, filter, "F")
   

   -- For each pixel, remove mean of kW x kH neighborhood
   local mid = math.ceil(self._kernel_size / 2)
   center:resizeAs(input):copy(input)
   center:add(-1, convout[{{},{mid, -mid},{mid, -mid}}])

   -- Scale down norm of kW x kH patch if norm is bigger than 1
   square:pow(center, 2)
   convout:conv2(square, filter, 'F')
   
   denom:resizeAs(input)
   denom:copy(convout[{{},{mid, -mid},{mid, -mid}}]) -- makes it contiguous
   denom:sqrt()
   -- per image mean : batchSize x 1
   mean:mean(denom:view(denom:size(1),-1), 2)
   divisor:gt(mean:view(mean:size(1),1,1):expandAs(denom), denom)
   divisor:apply(function(x) 
         return x>self._threshold and x or self._threshold 
      end)
   center:cdiv(divisor)
   
   return center
end
