function string.tomodule(modulename,splitter)
   splitter = splitter or '[.]'
   assert(type(modulename) == 'string')
   local modula = _G
   for i, name in ipairs(_.split(modulename,splitter)) do
      modula = modula[name] or require(modula)
   end
   return modula
end

--[[ From https://github.com/rosejn/lua-util: ]]--
-- Boolean predicate to determine if a path points to a valid file or directory.
function dp.is_file(path)
   return paths.filep(path) or paths.dirp(path)
end

-- Check that a data directory exists, and create it if not.
function dp.mkdir(dir)
   if not paths.dirp(dir) then
      paths.mkdir(dir)
   end
end
-- DEPRECATED : use dp.mkdir instead
dp.check_and_mkdir = dp.mkdir

-- Download the file at location url.
function dp.download_file(url)
   local protocol, scpurl, filename = url:match('(.-)://(.*)/(.-)$')
   if protocol == 'scp' then
       os.execute(string.format('%s %s %s', 'scp', scpurl .. '/' .. filename, filename))
   else
       os.execute('wget ' .. url)
   end
end

-- Temporarily changes the current working directory to call fn, 
-- returning its result.
function dp.do_with_cwd(path, fn)
   local cur_dir = lfs.currentdir()
   lfs.chdir(path)
   local res = fn()
   lfs.chdir(cur_dir)
   return res
end


-- If file doesn't exists at path, downloads it from url into path 
function dp.check_and_download_file(path, url)
   if not paths.filep(path) then
      local dirPath = paths.dirname(path)
      dp.mkdir(dirPath)
      dp.do_with_cwd(
         dirPath, 
         function() dp.download_file(url) end
      )
   end
   return path
end

-- Decompress a .tar, .tgz or .tar.gz file.
function dp.decompress_tarball(srcPath, dstPath)
   local dstPath = dstPath or '.'
   paths.mkdir(dstPath)
   if srcPath:match("%.tar$") then
      os.execute('tar -xvf ' .. srcPath .. ' -C ' .. dstPath)
   else
      os.execute('tar -xvzf ' .. srcPath .. ' -C ' .. dstPath)
   end
end

-- unzip a .zip file
function dp.unzip(srcPath, dstPath)
   local dstPath = dstPath or '.'
   paths.mkdir(dstPath)
   os.execute('unzip ' .. srcPath .. ' -d ' .. dstPath)
end

-- gunzip a .gz file
function dp.gunzip(srcPath, dstPath)
   assert(not dstPath, "destination path not supported with gunzip")
   os.execute('gunzip ' .. srcPath)
end

function dp.decompress_file(srcPath, dstPath)
    if string.find(srcPath, ".zip") then
        dp.unzip(srcPath, dstPath)
    elseif string.find(srcPath, ".tar") or string.find(srcPath, ".tgz") then
        dp.decompress_tarball(srcPath, dstPath)
    elseif string.find(srcPath, ".gz") or string.find(srcPath, ".gzip") then
        dp.gunzip(srcPath, dstPath)
    else
        print("Don't know how to decompress file: ", srcPath)
    end
end

--[[ End From ]]--

function dp.printG()
   for k,v in pairs(_.omit(_G, 'torch', 'paths', 'nn', 'xlua', '_', 
                           'underscore', 'io', 'utils', '_G', 'nnx', 
                           'optim', '_preloaded_ ', 'math', 'libfs',
                           'cutorch', 'image')) do
      print(k, type(v))
   end
end

function dp.distReport(dist, sort_dist)
   local dist = torch.div(dist, dist:sum()+0.000001)
   local report = {
      dist=dist, min=dist:min(), max=dist:max(),   
      mean=dist:mean(), std=dist:std()
   }
   if sort_dist then
      report.dist = dist:sort()
   end
   return report
end
   
function dp.reverseDist(dist, inplace)
   local reverse = dist
   if not inplace then
      reverse = dist:clone()
   end
   if dist:dim() == 1 then
      -- reverse distribution and make unlikely values more likely
      reverse:add(-reverse:max()):mul(-1):add(dist:min())
      reverse:div(math.max(reverse:sum(),0.000001))
   elseif dist:dim() == 2 then
      -- reverse distribution and make unlikely values more likely
      reverse:add(reverse:max(2):mul(-1):resize(reverse:size(1),1):expandAs(reverse)):mul(-1):add(dist:min(2):resize(reverse:size(1),1):expandAs(reverse))
      reverse:cdiv(reverse:sum(2):add(0.000001):resize(reverse:size(1),1):expandAs(reverse))
   end
   return reverse
end


-- Generates a globally unique identifier.
-- If a namespace is provided it is concatenated with 
-- the time of the call, and the next value from a sequence
-- to get a pseudo-globally-unique name.
-- Otherwise, we concatenate the linux hostname
local counter = 1
function dp.uniqueID(namespace, separator)
   local separator = separator or ':'
   local namespace = namespace or os.hostname()
   local uid = namespace..separator..os.time()..separator..counter
   counter = counter + 1
   return uid
end

function list_iter (t)
   local i = 0
   local n = table.getn(t)
   return function ()
            i = i + 1
            if i <= n then return t[i] end
          end
end

function math.round(a)
   if (a - math.floor(a)) >= 0.5 then
      return math.ceil(a)
   end
   return math.floor(a)
end 

-- security risk, but useful for unpacking multi-table strings :
-- a = dp.returnString("{1,3,{5,7},{3,4}}")
function dp.returnString(str)
   return loadstring(" return "..str)()
end

function dp.vprint(verbose, str)
   if verbose then
      print(str)
   end
end

-- count files in paths
function dp.countFiles(pathList)
   pathList = (torch.type(pathList) == 'string') and {pathList} or pathList
   local nFile = 0
   for i,path in ipairs(pathList) do
      for imageFile in lfs.dir(path) do
         if imageFile ~= '..' and imageFile ~= '.' then
            nFile = nFile + 1
         end
      end
   end
   return nFile
end

-- takes paths to directories of images and tensorizes them;
-- images will be resized to fit into tensor of shape 'bchw';
-- images will be shuffled according to shuffle inputs tensor (if provided);
-- targets tensor will be used to store index of paths used (if provided).
function dp.images2tensor(inputs, targets, paths, shuffle, verbose)
   if targets and not torch.isTensor(targets) then
      verbose = shuffle
      shuffle = paths
      paths = targets
      targets = nil
   end
   paths = torch.type(paths) == 'string' and {paths} or paths
   local k = 1
   local buffer = torch.DoubleTensor()
   for i, path in ipairs(paths) do
      for imageFile in lfs.dir(path) do
         local img_path = paths.concat(path, imageFile)
         if imageFile ~= '..' and imageFile ~= '.' then
            local idx = shuffle and shuffle[k] or idx
            local img = image.load(img_path)
            buffer:resize(inputs:size(2), inputs:size(3), inputs:size(4))
            image.scale(buffer, img, 'bilinear')
            inputs[idx]:copy(buffer)
            k = k + 1
            if verbose then
               xlua.progress(k, inputs:size(1))
            end
            if targets then
               targets[idx] = i
            end
         end
         collectgarbage()
      end
   end
   return inputs, targets
end

function dp.reload(mod, ...)
    package.loaded[mod] = nil
    return require(mod, ...)
end

------------------------ Queue -----------------------------
local Queue = torch.class("dp.Queue")
function Queue:__init()
   self.first = 0
   self.last = -1
   self.list = {}
end

function Queue:put(value)
   local first = self.first - 1
   self.first = first
   self.list[first] = value
end

function Queue:empty()
   return self.first > self.last
end
 
function Queue:get()
   local last = self.last
   if self:empty() then 
      error("Queue is empty")
   end
   local value = self.list[last]
   self.list[last] = nil  
   self.last = last - 1
   return value
end
