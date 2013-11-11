require 'paths'
require 'torch'
require 'image'

require 'util'
require 'util/file'


------------------------------------------------------------------------
--[[ dp ]]--
-- dEEp learning library for torch 7, inspired by pylearn2.
------------------------------------------------------------------------


dp = {}

dp.TORCH_DIR = os.getenv('TORCH_DATA_PATH')
dp.DATA_DIR  = paths.concat(TORCH_DIR, 'data')

torch.include('dp', 'datatensor.lua')
torch.include('dp', 'dataset.lua')
torch.include('dp', 'datasource.lua')
torch.include('dp', 'preprocess.lua')
torch.include('dp', 'datasampler.lua')
torch.include('dp', 'learn.lua')






function dp.rand_between(min, max)
   return math.random() * (max - min) + min
end


function dp.rand_pair(v_min, v_max)
   local a = dp.rand_between(v_min, v_max)
   local b = dp.rand_between(v_min, v_max)
   return a,b
end


function dp.sort_by_class(samples, labels)
    local size = labels:size()[1]
    local sorted_labels, sort_indices = torch.sort(labels)
    local sorted_samples = samples.new(samples:size())

    for i=1, size do
        sorted_samples[i] = samples[sort_indices[i]]
    end

    return sorted_samples, sorted_labels
end


function dp.rotator(start, delta)
   local angle = start
   return function(src, dst)
      image.rotate(dst, src, angle)
      angle = angle + delta
   end
end


function dp.translator(startx, starty, dx, dy)
   local started = false
   local cx = startx
   local cy = starty
   return function(src, dst)
      image.translate(dst, src, cx, cy)
      cx = cx + dx
      cy = cy + dy
   end
end


function dp.zoomer(start, dz)
   local factor = start
   return function(src, dst)
      local src_width  = src:size(2)
      local src_height = src:size(3)
      local width      = math.floor(src_width * factor)
      local height     = math.floor(src_height * factor)

      local res = image.scale(src, width, height)
      if factor > 1 then
         local sx = math.floor((width - src_width) / 2)+1
         local sy = math.floor((height - src_height) / 2)+1
         dst:copy(res:narrow(2, sx, src_width):narrow(3, sy, src_height))
      else
         local sx = math.floor((src_width - width) / 2)+1
         local sy = math.floor((src_height - height) / 2)+1
         dst:zero()
         dst:narrow(2, sx,  width):narrow(3, sy, height):copy(res)
      end

      factor = factor + dz
   end
end

