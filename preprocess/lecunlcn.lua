-----------------------------------------------------------------------
--[[ LeCunLCN ]]--
-- Performs Local Contrast Normalization on images
-- http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
-----------------------------------------------------------------------
local LeCunLCN = torch.class("dp.LeCunLCN", "dp.Preprocess")
LeCunLCN.isLeCunLCN = true

function LeCunLCN:__init(config)
   local args, batch_size
   args, self._kernel_size, self._threshold, batch_size, self._channels,
      self._progress = xlua.unpack(
      {config},
      'LeCunLCN', 
      'LeCunLCN constructor',
      {arg='kernel_size', type='number', default=9, 
       help='local contrast kernel size'},
      {arg='threshold', type='number', default=1e-4,
       help='threshold for denominator'},
      {arg='batch_size', type='number', default=1024,
       help='batch_size used for performing the preprocessing'},
      {arg='channels', type='table',
       help='List of channels to normalize. Defaults to {1,2,3}'},
      {arg='progress', type='boolean', default=true, 
       help='display progress bar'}
   )
   self._sampler = dp.Sampler{batch_size = batch_size}
   self._channels = self._channels or {1,2,3}
   self._filter = self:_gaussian_filer(self._kernel_size)
   -- buffers
   self._convout = torch.Tensor()
   self._center = torch.Tensor()
   self._square = torch.Tensor()
   self._mean = torch.Tensor()
   self._divisor = torch.Tensor()
   self._result = torch.Tensor()
end

function LeCunLCN:_gaussian_filter(kernel_size)
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
   print('Start LeCunLCN Preprocessing ... ')
   
   local batch, i, n, last_n
   local n_batch = 1
   local sampler = self._sampler:sampleEpoch(dv)
   while true do
      -- reuse the batch object
      batch, i, n = sampler(batch)
      if (not batch) and self._progress then 
         -- for aesthetics :
         xlua.progress(last_n, last_n)
         break 
      end
      self:_transform(batch:inputs())
      if self._progress then
         -- disp progress
         xlua.progress(i, n)
      end
      last_n = n
      n_batch = n_batch + 1
   end

end

-- expects 'bhw' input
function LeCunLCN:normalize(input)   
   local filter, convout = self._filter, self._convout
   local center, square = self._center, self._square
   local mean, divisor = self._mean = self._divisor
   filter = filter:view(1,filter:size(1),filter:size(2)):expandAs(input)
   convout:conv2(input, filter, "F")
   

   -- For each pixel, remove mean of kW x kH neighborhood
   local mid = math.ceil(self._kernel_size / 2)
   center:resizeAs(input):copy(input)
   center:add(-1, convout[{{},{mid, -mid},{mid, -mid}}])

   -- Scale down norm of kW x kH patch if norm is bigger than 1
   square:pow(center, 2)
   convout:conv2(square, filter, 'F')
   
   denom = convout[{{},{mid, -mid},{mid, -mid}}]
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

-- expects x to have view 'bhwc'
function LeCunLCN:transform(x)
   if torch.type(x) ~= torch.type(self._filter) then
      self._filter = self._filter:type(torch.type(x))
      self._convout = self._convout:type(torch.type(x))
      self._center = self._center:type(torch.type(x))
      self._square = self._square:type(torch.type(x))
      self._mean = self._mean:type(torch.type(x))
      self._divisor = self._divisor:type(torch.type(x))
      self._result = self._result:type(torch.type(x))
   end
   
   self._result:resizeAs(x):copy(x)
   for i,channelIdx in ipairs(self._channels) do
      assert(torch.type(channelIdx) == 'number') 
      assert(channelIdx >= 0 and channelIdx <= x:size(4))

      self._result:select(4,channelIdx) = self:normalize(x:select(4,channelIdx))
   end
   return self._result
end

function LeCunLCN:apply(dv, can_fit)
   local batch, i, n, last_n
   local n_batch = 1
   local sampler = self._sampler:sampleEpoch(dv)
   
   while true do
      -- reuse the batch object
      batch, i, n = sampler(batch)
      if (not batch) and self._progress then 
         -- for aesthetics :
         xlua.progress(last_n, last_n)
         break 
      end
      
      local view = batch:inputs()
      view:replace("bhwc", 
         self:transform(
            view:forward("bhwc")
         )
      )
      
      if self._progress then
         -- disp progress
         xlua.progress(i, n)
      end
      last_n = n
      n_batch = n_batch + 1
   end
end
