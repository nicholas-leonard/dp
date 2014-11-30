-----------------------------------------------------------------------
--[[ LeCunLCN ]]--
-- Performs Local Contrast Normalization on images
-- http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
-- You should probably use dp.GCN before applying LeCunLCN to
-- mitigate border effects.
-----------------------------------------------------------------------
local LeCunLCN = torch.class("dp.LeCunLCN", "dp.Preprocess")
LeCunLCN.isLeCunLCN = true

function LeCunLCN:__init(config)
   config = config or {}
   local args
   args, self._kernel_size, self._kernel_std, self._threshold, 
      self._divide_by_std, self._batch_size, self._channels, 
      self._progress = xlua.unpack(
      {config},
      'LeCunLCN', 
      'LeCunLCN constructor',
      {arg='kernel_size', type='number', default=9, 
       help='gaussian kernel size. Should be an odd number.'},
      {arg='kernel_std', type='number', default=2, 
       help='standard deviation of gaussian kernel.'..
       'Higher values remove lower frequency features,'..
       'i.e. a value of infinity will have no effect.'},
      {arg='threshold', type='number', default=1e-4,
       help='threshold for denominator'},
      {arg='divide_by_std', type='boolean', default=false,
       help='instead of divisive normalization, divide by std'},
      {arg='batch_size', type='number', default=256,
       help='batch_size used for performing the preprocessing'},
      {arg='channels', type='table',
       help='List of channels to normalize. Defaults to {1,2,3}'},
      {arg='progress', type='boolean', default=true, 
       help='display progress bar'}
   )
   assert(self._kernel_size % 2 == 1, "kernel_size should be odd (not even)")
   self._sampler = dp.Sampler{batch_size=self._batch_size}
   self._channels = self._channels or {1,2,3}
   self._filter = self.gaussianFilter(self._kernel_size, self._kernel_std)
   -- buffers
   self._convout = torch.Tensor()
   self._center = torch.Tensor()
   self._square = torch.Tensor()
   self._mean = torch.Tensor()
   self._divisor = torch.Tensor()
   self._result = torch.Tensor()
   self._denom = torch.Tensor()
   self._largest = torch.Tensor()
   self._indice = torch.LongTensor()
end

-- static method
function LeCunLCN.gaussianFilter(kernel_size, kernel_std)
   local x = torch.zeros(kernel_size, kernel_size)
   local _gauss = function(x, y, sigma)
      local Z = 2 * math.pi * math.pow(sigma, 2)
      return 1 / Z * math.exp(-(math.pow(x,2)+math.pow(y,2))/(2 * math.pow(sigma,2)))
   end
   
   local mid = math.ceil(kernel_size / 2)
   for i = 1, kernel_size do
      for j = 1, kernel_size do
         x[i][j] = _gauss(i-mid, j-mid, kernel_std)
      end
   end
   
   return x / x:sum()
end

function LeCunLCN:apply(dv, can_fit)   
   if self._progress then
      print"applying LeCunLCN preprocessing"
   end
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
      self._largest = self._largest:type(torch.type(x))
   end
   
   self._result:resizeAs(x):copy(x)
   for i,channelIdx in ipairs(self._channels) do
      assert(torch.type(channelIdx) == 'number') 
      assert(channelIdx >= 0)
      
      if channelIdx > x:size(4) then
        break
      end

      self._result:select(4,channelIdx):copy(self:normalize(x:select(4,channelIdx)))
   end
   return self._result
end

-- expects input to have view 'bhw'
function LeCunLCN:normalize(input)   
   local filter, convout = self._filter, self._convout
   local center, square = self._center, self._square
   local mean, divisor = self._mean, self._divisor
   local denom, indice, largest = self._denom, self._indice, self._largest
   
   --[[ subtractive normalization ]]--
   filter = filter:view(1, filter:size(1), filter:size(2))
   filter = filter:expand(input:size(1), filter:size(2), filter:size(3))
   convout:conv2(input, filter, "F")
   
   -- For each pixel, remove mean of kW x kH neighborhood
   local mid = math.ceil(self._kernel_size / 2)
   center:resizeAs(input):copy(input)
   center:add(-1, convout[{{},{mid, -mid},{mid, -mid}}])

   --[[ divisive normalization ]]--
   if self._divide_by_std then
      -- divide by standard deviation of each image
      denom:std(center:view(center:size(1), -1), 2):add(self._threshold)
      center:cdiv(denom:view(denom:size(1), 1, 1):expandAs(center))
      return center
   end
   
   -- Scale down norm of kW x kH patch if norm is bigger than 1
   square:pow(center, 2)
   convout:conv2(square, filter, 'F')
   
   denom:resizeAs(input)
   denom:copy(convout[{{},{mid, -mid},{mid, -mid}}]) -- makes it contiguous
   denom:sqrt()
   -- per image mean : batchSize x 1
   mean:mean(denom:view(denom:size(1),-1), 2)
   largest:resize(denom:size(1), denom:size(2), denom:size(3), 2)
   largest:select(4,1):copy(denom)
   largest:select(4,2):copy(mean:view(mean:size(1),1,1):expandAs(denom))
   divisor:max(indice, largest, 4)
   divisor:apply(function(x) 
         return x>self._threshold and x or self._threshold 
      end)
   center:cdiv(divisor)
   
   return center
end
