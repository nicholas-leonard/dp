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
      -- return a table of key, value = {'axes label', 'tensor size for that label'}
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

   local new_filters
   
   if #filters_axes == 2 then
      local new_filters_axes = {_.indexOf(filters_axes, 'h'),
                                _.indexOf(filters_axes, 'w')}
                                
      new_filters = table.eq(filters_axes, {'h', 'w'}) 
                            and filters
                            or filters:transpose(1, 2)
   
      new_filters = filters:repeatTensor(images_size.c, 1, 1)
      
   elseif #filters_axes == 3 then
      local new_filters_axes = {_.indexOf(filters_axes, 'c'),
                                _.indexOf(filters_axes, 'h'),
                                _.indexOf(filters_axes, 'w')}
                                
      new_filters = table.eq(filters_axes, {'c', 'h', 'w'}) 
                            and filters
                            or torch.swapaxes(filters, new_filters_axes)
   else
      error('filters dim is not 2 or 3')
   end

   
   if #images_axes == 3 then
   
      local resized_images = table.eq(images_axes, {'c', 'h', 'w'})
                             and images
                             or torch.swapaxes(images, {_.indexOf(images_axes, 'c'),
                                                        _.indexOf(images_axes, 'h'),
                                                        _.indexOf(images_axes, 'w')})

      local conv_images = torch.conv2(resized_images, new_filters, border_mode)
      
      -- swap back to the original axes
      return torch.swapaxes(conv_images, {[_.indexOf(images_axes, 'c')] = 1,
                                                [_.indexOf(images_axes, 'h')] = 2,
                                                [_.indexOf(images_axes, 'w')] = 3})

   elseif #images_axes == 4 then      
      local conv_images = border_mode == 'F' 
                                       and 
                                       torch.zeros(images_size.b, images_size.c, 
                                       images_size.h + filters_size.h - 1, 
                                       images_size.w + filters_size.w - 1)
                                       or
                                       torch.zeros(images_size.b, images_size.c, 
                                       images_size.h - filters_size.h + 1, 
                                       images_size.w - filters_size.w + 1)
                               
      local resized_images = table.eq(images_axes, {'b', 'c', 'h', 'w'})
                             and images
                             or torch.swapaxes(images, {_.indexOf(images_axes, 'b'),
                                                        _.indexOf(images_axes, 'c'),
                                                        _.indexOf(images_axes, 'h'),
                                                        _.indexOf(images_axes, 'w')})
      
      for b = 1, images_size.b do
         conv_images[b] = torch.conv2(resized_images[b], new_filters, border_mode)
      end

      -- swap back to the original axes
      return torch.swapaxes(conv_images, {[_.indexOf(images_axes, 'b')] = 1,
                                          [_.indexOf(images_axes, 'c')] = 2,
                                          [_.indexOf(images_axes, 'h')] = 3,
                                          [_.indexOf(images_axes, 'w')] = 4})
   else
      error('images dim is not 3 or 4')
   end
end
   
   