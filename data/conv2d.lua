require 'torch'

-- wrapper for torch.conv2 to make it compatible with dp axes label
function conv2d(image, filters, border_mode)

   -- image is 4D double tensor of axes {'b', 'h', 'w', 'c'}
   -- filters can be 3D tensor of axes {'h', 'w', 'c'} or 2D tensor of axes {'h', 'w'}
   -- 2D filters by default applies to all channels
   -- border_mode:
   -- 'valid' : only apply filter to complete patches of the image. Generates
   --           output of shape: image_shape - filter_size + 1
   -- 'full'  : zero-pads image to multiple of filter shape to generate output
   --           of shape: image_shape + filter_size - 1
   
   
   border_mode = border_mode or 'full'
   assert(image:dim() == 4, 'image ~= 4')
   local image_shape = {b = image:size(1),
                        h = image:size(2), 
                        w = image:size(3), 
                        c = image:size(4)}
   
   local filter_shape = {}
   if filters:dim() == 2 then
      filter_shape = {h = filters:size(1), w = filters:size(2)}
   elseif filters:dim() == 3 then
      filter_shape = {h = filters:size(1), w = filters:size(2), c = filters:size(3)}
   else
      error ('filter dim is not 2 or 3')
   end


   local _conv2 = function(images, 3d_filter, border_mode)
      local t = torch.zeros(image_shape['b'], image_shape['h'], 
                            image_shape['w'], image_shape['c'])
      
      for b = 1, image_shape.b do
         local new_image = images[b]:resize(image_shape.c, image_shape.h, image_shape.w)
         t[b] = torch.conv2(new_image, filters, border_mode):resize(image_shape.h, 
                                                                    image_shape.w,
                                                                    image_shape.c)
         end
      end 
      
      return t
   end
                    
   
   -- no zero padding, return image size of image_size - filter_size + 1
   if border_mode == 'valid' then
      
      if filters:dim() == 2 then
         filters:repeatTensor(image_shape.c, 1, 1)
         return _conv2(image, filters, 'V')
         
      elseif filters:dim() == 3 then
         filters:resize(image_shape.c, image_shape.h, image_shape.w)
         return _conv2(image, filters, 'V')
      end
   
   -- with zero paddings, return image size of image_size + filter_size + 1
   elseif border_mode == 'full' then
      if filters:dim() == 2 then
         filters:repeatTensor(image_shape.c, 1, 1)
         return _conv2(image, filters, 'F')
         
      elseif filters:dim() == 3 then
         filters:resize(image_shape.c, image_shape.h, image_shape.w)
         return _conv2(image, filters, 'F')
      end
      
   else
      error('border_mode is not set to \'valid\' or \'full\'')
   
   end
   
end
   
   