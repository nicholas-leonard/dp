require 'torch'
require 'image'
function dp.conv2d(images, filters, images_axes, filters_axes, border_mode)

   -- images:
   --    3D or 4D double tensor with axes given by images_axes
   -- filters:
   --    2D or 3D double tensor with axes given by filters_axes
   -- images_axes:
   --    3D table with labels (in any order) in {'h', 'w', 'c'} or 
   --    4D table with labels (in any order) in {'b', 'h', 'w', 'c'} 
   -- filter_axes:
   --    2D table with labels (in any order) in {'h', 'w'} or
   --    3D table with labels (in any order) in {'h', 'w', 'c'}
   -- 2D filters by default applies to all channels of the images
   -- border_mode:
   -- 'V' : only apply filter to complete patches of the image. Generates
   --       output of shape: image_shape - filter_shape + 1
   -- 'F' : zero-pads image to multiple of filter shape to generate output
   --       of shape: image_shape + filter_shape - 1
   -- return:
   --    A convolved double tensor of same size and axes as input images
   
   local _size = function(tensor, axes)
      -- return a table of key, value = 'axes label', 'tensor size for that label'
      t = {}
      for k, v in ipairs(axes) do
         t[v] = tensor:size(k)
      end
      return t
   end         
 
   border_mode = border_mode or 'F'
   assert(#images_axes == images:dim(), 'Error: images dim is not equal to images_axes dim')
   assert(#filters_axes == filters:dim(), 'Error: filters dim is not equal to filters_axes dim')
   local images_size = _size(images, images_axes)
   local filters_size = _size(filters, filters_axes)
   local orig_img_size = images:size()
   local orig_fil_size = filters:size()

   if #images_axes == 3 then
   
      local _conv2 = function(filters)
         if not table.eq(images_axes, {'c', }) then
         images:resize(images_size.c, images_size.h, images_size.w)
         end
         local orig_axes = {_.indexOf(images_axes, 'c'), _.indexOf(images_axes, 'h'),
                           _.indexOf(images_axes, 'w')}
         local new_images = torch.swapaxes(torch.conv2(images, filters, border_mode), orig_axes)
         images:resize(orig_img_size)
         return new_images
      end
         
      if #filters_axes == 2 then
         return _conv2(filters:repeatTensor(images_size.c, 1, 1))
         
      elseif #filters_axes == 3 then
         local new_images = _conv2(filters:resize(images_size.c, images_size.h, images_size.w)) 
         filters:resize(orig_fil_size)
         return new_images      
      else
         error('filters dim is not 2 or 3')
      end
   
   elseif #images_axes == 4 then      
      local new_images = border_mode == 'F' 
                                       and 
                                       torch.zeros(images_size.b, images_size.c, 
                                       images_size.h + filters_size.h - 1, 
                                       images_size.w + filters_size.w - 1)
                                       or
                                       torch.zeros(images_size.b, images_size.c, 
                                       images_size.h - filters_size.h + 1, 
                                       images_size.w - filters_size.w + 1)
      
      local _conv2 = function(filters)
         images:resize(images_size.b, images_size.c, images_size.h, images_size.w)
         for b = 1, images_size.b do
            new_images[b] = torch.conv2(images[b], filters, border_mode)
         end
         local orig_axes = {_.indexOf(images_axes, 'b'), _.indexOf(images_axes, 'c'),
                            _.indexOf(images_axes, 'h'), _.indexOf(images_axes, 'w')}
         images:resize(orig_img_size)  
         torch.swapaxes(new_images, orig_axes)
      end
                                                   
      if #filters_axes == 2 then
         _conv2(filters:repeatTensor(images_size.c, 1, 1))
         return new_images
      
      elseif #filters_axes == 3 then
         _conv2(filters:resize(filters_size.c, filters_size.h, filters_size.w))
         filters:resize(orig_fil_size)
         return new_images
      
      else
         error('filters dim is not 2 or 3')
      end
   
   else
      error('images dim is not 3 or 4')
   end
end
   
   