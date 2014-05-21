------------------------------------------------------------------------
--[[ ImageView ]]-- 
-- A View holding a tensor of images.
------------------------------------------------------------------------
local ImageView, parent = torch.class("dp.ImageView", "dp.DataView")
ImageView.isImageView = true

-- batch x height x width x channels/colors
function ImageView:bhwc()
   if self._view == 'bhwc' then
      return nn.Identity()
   end
   return self:transpose('bhwc')
end

-- View used by SpacialConvolutionCUDA
function ImageView:chwb()
   if self._view == 'chwb' then
      return nn.Identity()
   end
   return self:transpose('chwb')
end

-- View used by SpacialConvolution
function ImageView:bchw()
   if self._view == 'bchw' then
      return nn.Identity()
   end
   return self:transpose('bchw')
end
