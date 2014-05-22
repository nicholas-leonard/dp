-----------------------------------------------------------------------
--[[ LeCunLCN ]]--
-- Performs Local Contrast Normalization on images
-- http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
-----------------------------------------------------------------------
local LeCunLCN = torch.class("dp.LeCunLCN", "dp.Preprocess")
LeCunLCN.isLeCunLCN = true

function LeCunLCN:__init(...)
   local args
   args, self._kernel_size, self._threshold = xlua.unpack(
      {... or {}},
      'LeCunLCN', 'LeCunLCN constructor',
      {arg='kernel_size', type='number', 
       help=[[local contrast kernel size]], default=9},
      {arg='threshold', type='number',
       help=[[threshold for denominator]], default=1e-4}
   )     
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
   print ('Start LeCunLCN Preprocessing ... ')
   local data = dv:forward('bhwc')
          
   local filters = self:_gaussian_filter(self._kernel_size)
   print('start convolving data')
   local convout = dp.conv2d(data, filters, {'b','h','w','c'}, {'h', 'w'})
   print('1/3 finished convolving data, start convolving centered_X ...')
   local mid = math.ceil(self._kernel_size / 2)
   local centered_X = data - convout[{{1, -1},{mid, -mid},{mid, -mid},{1, -1}}]
   local sum_sqr_XX = dp.conv2d(torch.pow(centered_X,2), filters, 
                                {'b', 'h', 'w', 'c'}, {'h', 'w'})
   print('2/3 finished convolving centered_X, start finding divisor ...')
   local denom = torch.sqrt(sum_sqr_XX[{{1,-1},{mid, -mid},{mid, -mid},{1, -1}}])
   local resized = denom:resize(denom:size(1), denom:size(2) * denom:size(3), denom:size(4))
   local per_img_mean = torch.mean(resized, 2)
   per_img_mean:resize(per_img_mean:size(1), 1, 1, per_img_mean:size(3))
   local expanded = per_img_mean:expandAs(data)
   denom:resize(data:size(_.indexOf(axes, 'b')), data:size(_.indexOf(axes, 'h')),
                data:size(_.indexOf(axes, 'w')), data:size(_.indexOf(axes, 'c')))
   local divisor = denom:map(expanded, function(d, e) return d>e and d or e end)
   print('3/3 finished finding divisor, finishing preprocessing ..')
   divisor = divisor:apply(function(x) return x>self._threshold and x or self._threshold end)
   print ('LeCunLCN Preprocessing completed')
   local new_X = centered_X:cdiv(divisor)
   dv:replace('bhwc', new_X)
end
