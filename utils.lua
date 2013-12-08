require 'torch'
require 'string'
_ = require 'underscore'
require 'xlua'
require 'fs'
require 'paths'
require 'os'
require 'sys'

--useful for validating if an object is an instance of a class, 
--even when the class is a super class.
--e.g pattern = "^torch[.]%a*Tensor$"
--typepattern(torch.Tensor(4), pattern) and
--typepattern(torch.DoubleTensor(5), pattern) are both true.
--typepattern(3, pattern)
function typepattern(obj, pattern)
   local class = type(obj)
   if class == 'userdata' then
      class = torch.typename(obj)
   end
   local match = string.match(class, pattern)
   if match == nil then
      match = false
   end
   return match
end

function torch.isTensor(obj)
   return typepattern(obj, "^torch[.]%a*Tensor$")
end

function list_iter (t)
   local i = 0
   local n = table.getn(t)
   return function ()
            i = i + 1
            if i <= n then return t[i] end
          end
end

--http://lua-users.org/wiki/TableUtils
function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

function table.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end

--http://stackoverflow.com/questions/8722620/comparing-two-index-tables-by-index-value-in-lua
local function recursive_compare(t1,t2)
  -- Use usual comparison first.
  if t1==t2 then return true end
  -- We only support non-default behavior for tables
  if (type(t1)~="table") and (type(t2)~="table") then return false end
  -- They better have the same metatables
  local mt1 = getmetatable(t1)
  local mt2 = getmetatable(t2)
  if( not recursive_compare(mt1,mt2) ) then return false end

  -- Check each key-value pair
  -- We have to do this both ways in case we miss some.
  -- TODO: Could probably be smarter and not check those we've 
  -- already checked though!
  for k1,v1 in pairs(t1) do
    local v2 = t2[k1]
    if( not recursive_compare(v1,v2) ) then return false end
  end
  for k2,v2 in pairs(t2) do
    local v1 = t1[k2]
    if( not recursive_compare(v1,v2) ) then return false end
  end

  return true  
end
table.eq = recursive_compare


--[[ From https://github.com/rosejn/lua-util: ]]--

-- Boolean predicate to determine if a path points to a valid file or directory.
function is_file(path)
    return paths.filep(path) or paths.dirp(path)
end

-- Check that a data directory exists, and create it if not.
function check_and_mkdir(dir)
  if not paths.filep(dir) then
    fs.mkdir(dir)
  end
end


-- Download the file at location url.
function download_file(url)
    local protocol, scpurl, filename = url:match('(.-)://(.*)/(.-)$')
    if protocol == 'scp' then
        os.execute(string.format('%s %s %s', 'scp', scpurl .. '/' .. filename, filename))
    else
        os.execute('wget ' .. url)
    end
end


-- Temporarily changes the current working directory to call fn, 
-- returning its result.
function do_with_cwd(path, fn)
    local cur_dir = fs.cwd()
    fs.chdir(path)
    local res = fn()
    fs.chdir(cur_dir)
    return res
end


-- Check that a file exists at path, and if not downloads it from url.
function check_and_download_file(path, url)
  if not paths.filep(path) then
      do_with_cwd(paths.dirname(path), function() download_file(url) end)
  end

  return path
end

function decompress_file(path)
    if string.find(path, ".zip") then
        unzip(path)
    elseif string.find(path, ".tar.gz") or string.find(path, ".tgz") then
        decompress_tarball(path)
    elseif string.find(path, ".gz") or string.find(path, ".gzip") then
        gunzip(path)
    else
        print("Don't know how to decompress file: ", path)
    end
end

--[[ End From ]]--

-- From http://stackoverflow.com/questions/1283388/lua-merge-tables
function merge(t1, t2)
    for k, v in pairs(t2) do
        if (type(v) == "table") and (type(t1[k] or false) == "table") then
            merge(t1[k], t2[k])
        else
            t1[k] = v
        end
    end
    return t1
end
table.merge = merge


function constrain_norms(max_norm, axis, matrix)
   local norms = torch.norm(weight,2,axis)
   -- clip
   local new_norms = norms:clone()
   new_norms[torch.gt(norms, max_norm)] = max_norm
   local div = torch.cdiv(new_norms, torch.add(norms,1e-7))
   matrix:cmul(div:expandAs(matrix))
end

function typeString_to_tensorType(type_string)
   if type_string == 'cuda' then
      return 'torch.CudaTensor'
   elseif type_string == 'float' then
      return 'torch.FloatTensor'
   elseif type_string == 'double' then
      return 'torch.DoubleTensor'
   end
end
