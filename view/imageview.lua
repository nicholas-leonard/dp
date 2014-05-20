------------------------------------------------------------------------
--[[ ImageView ]]-- 
-- A View holding a tensor of images.
------------------------------------------------------------------------
local ImageView, parent = torch.class("dp.ImageView", "dp.View")
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

-- a generic function for transposing images
function ImageView:transpose(new_view)
   local view = _.split(self._view)
   local transpositions = {}
   for i=1,4 do
      local j = _.indexOf(view, new_view:sub(i,i))
      if i ~= j then
         local char = view[i]
         view[i] = view[j]
         view[j] = char
         table.insert(transpositions, {j, i})
      end
   end
   return nn.Transpose(unpack(transpositions))
end

