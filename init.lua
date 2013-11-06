require 'paths'
require 'torch'
require 'image'

require 'util'
require 'util/file'

TORCH_DIR = os.getenv('TORCH_DATA_PATH')
DATA_DIR  = paths.concat(TORCH_DIR, 'data')

dataset = {}
-- Check locally and download dataset if not found.  Returns the path to the
-- downloaded data file.
function dataset.get_data(name, url, data_dir)
   local dset_dir   = paths.concat(data_dir or DATA_DIR, name)
   local data_file = paths.basename(url)
   local data_path = paths.concat(dset_dir, data_file)

   print("checking for file located at: ", data_path)

   check_and_mkdir(TORCH_DIR)
   check_and_mkdir(DATA_DIR)
   check_and_mkdir(dset_dir)
   check_and_download_file(data_path, url)

   return data_path
end


-- Downloads the data if not available locally, and returns local path.
function dataset.data_path(name, url, file, data_dir)
    local data_path  = dataset.get_data(name, url, data_dir)
    local data_dir   = paths.dirname(data_path)
    local local_path = paths.concat(data_dir, file)

    if not is_file(local_path) then
        do_with_cwd(data_dir,
          function()
              print("decompressing file: ", data_path)
              decompress_file(data_path)
          end)
    end

    return local_path
end


function dataset.scale(data, min, max)
    local range = max - min
    local dmin = data:min()
    local dmax = data:max()
    local drange = dmax - dmin

    data:add(-dmin)
    data:mul(range)
    data:mul(1/drange)
    data:add(min)
end


function dataset.rand_between(min, max)
   return math.random() * (max - min) + min
end


function dataset.rand_pair(v_min, v_max)
   local a = dataset.rand_between(v_min, v_max)
   local b = dataset.rand_between(v_min, v_max)
   return a,b
end


function dataset.sort_by_class(samples, labels)
    local size = labels:size()[1]
    local sorted_labels, sort_indices = torch.sort(labels)
    local sorted_samples = samples.new(samples:size())

    for i=1, size do
        sorted_samples[i] = samples[sort_indices[i]]
    end

    return sorted_samples, sorted_labels
end


function dataset.rotator(start, delta)
   local angle = start
   return function(src, dst)
      image.rotate(dst, src, angle)
      angle = angle + delta
   end
end


function dataset.translator(startx, starty, dx, dy)
   local started = false
   local cx = startx
   local cy = starty
   return function(src, dst)
      image.translate(dst, src, cx, cy)
      cx = cx + dx
      cy = cy + dy
   end
end


function dataset.zoomer(start, dz)
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

