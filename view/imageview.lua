------------------------------------------------------------------------
--[[ ImageView ]]-- 
-- A View holding a tensor of images.
------------------------------------------------------------------------
local ImageView, parent = torch.class("dp.ImageView", "dp.View")
ImageView.isImageView = true

-- batch feature
function View:bf()
   local view, data, dim = self._view, self._input, self._dim
   local b_pos = self:findAxis('b', view)
   -- was b
   if dim == 1 then
      if self._warn then
         print("View:feature Warning: provided data has one "..
               "dim. Assuming dim is 'b' axis with feature size of 1")
      end
      return nn.Reshape(1)
   end
   -- was b...
   local modula
   if b_pos ~= 1 then
      modula = nn.Transpose({1, b_pos})
   end
   if dim > 2 then
      local transpose = modula
      local reshape = nn.Reshape(self:sampleSize(b_pos))
      if transpose then
         modula = nn.Sequential()
         modula:add(transpose)
         modula:add(reshape)
      else
         modula = reshape
      end
   end
   return modula or nn.Identity()
end

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
   local view = self._view
   local transpositions = {}
   for i=1,4 do
      local j = self:findAxis(new_view[i], view)
      table.insert(transpositions, {j, i})
   end
   return nn.Transpose(unpack(transpositions))
end


