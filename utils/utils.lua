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
   local cur_dir = fs.cwd()
   fs.chdir(path)
   local res = fn()
   fs.chdir(cur_dir)
   return res
end


-- If file doesn't exists at path, downloads it from url into path 
function dp.check_and_download_file(path, url)
   if not paths.filep(path) then
      dp.do_with_cwd(
         paths.dirname(path), 
         function() dp.download_file(url) end
      )
   end
   return path
end

-- Decompress a .tgz or .tar.gz file.
function dp.decompress_tarball(path)
   os.execute('tar -xvzf ' .. path)
end

-- unzip a .zip file
function dp.unzip(path)
   os.execute('unzip ' .. path)
end

-- gunzip a .gz file
function dp.gunzip(path)
   os.execute('gunzip ' .. path)
end

function dp.decompress_file(path)
    if string.find(path, ".zip") then
        dp.unzip(path)
    elseif string.find(path, ".tar.gz") or string.find(path, ".tgz") then
        dp.decompress_tarball(path)
    elseif string.find(path, ".gz") or string.find(path, ".gzip") then
        dp.gunzip(path)
    else
        print("Don't know how to decompress file: ", path)
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
