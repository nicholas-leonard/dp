
--http://lua-users.org/wiki/TableUtils
function table.val_to_str ( v )
   if "string" == type( v ) then
      v = string.gsub( v, "\n", "\\n" )
      if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
         return "'" .. v .. "'"
      end
      return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
   else
      return "table" == type( v ) and table.tostring( v ) or tostring( v )
   end
end

function table.key_to_str ( k )
   if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
      return k
   else
      return "[" .. table.val_to_str( k ) .. "]"
   end
end

function table.tostring(tbl, newline, sort)
   local result, done = {}, {}
   for k, v in ipairs( tbl ) do
      table.insert( result, table.val_to_str( v ) )
      done[ k ] = true
   end
   local s = "="
   if newline then
      s = " : "
   end
   for k, v in pairs( tbl ) do
      if not done[ k ] then
         local line = table.key_to_str( k ) .. s .. table.val_to_str( v )
         table.insert(result, line)
      end
   end
   if sort then
      _.sort(result)
   end
   local res
   if newline then
      res = "{\n   " .. table.concat( result, "\n   " ) .. "\n}"
   else
      res = "{" .. table.concat( result, "," ) .. "}"
   end
   return res
end

function table.print(tbl)
   assert(torch.type(tbl) == 'table', "expecting table")
   print(table.tostring(tbl, true, true))
end

--http://stackoverflow.com/questions/2705793/how-to-get-number-of-entries-in-a-lua-table
function table.length(T)
   local count = 0
   for _ in pairs(T) do count = count + 1 end
   return count
end

--http://stackoverflow.com/questions/8722620/comparing-two-index-tables-by-index-value-in-lua
function table.eq(t1,t2)
   -- Use usual comparison first.
   if t1==t2 then return true end
   -- We only support non-default behavior for tables
   if (type(t1)~="table") and (type(t2)~="table") then return false end
   -- They better have the same metatables
   local mt1 = getmetatable(t1)
   local mt2 = getmetatable(t2)
   if( not table.eq(mt1,mt2) ) then return false end

   -- Check each key-value pair
   -- We have to do this both ways in case we miss some.
   -- TODO: Could probably be smarter and not check those we've 
   -- already checked though!
   for k1,v1 in pairs(t1) do
      local v2 = t2[k1]
      if( not table.eq(v1,v2) ) then return false end
   end
   for k2,v2 in pairs(t2) do
      local v1 = t1[k2]
      if( not table.eq(v1,v2) ) then return false end
   end

   return true  
end

-- recursively traverse a table:
function table.recurse(t1, t2, f)
   for k, v in pairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = table.recurse(t1[k] or {}, t2[k], f)
      else
         f(t1, k, v)
      end
   end
   return t1
end

-- From http://stackoverflow.com/questions/1283388/lua-merge-tables
-- values in table 1 have precedence
function table.merge(t1, t2)
    for k, v in pairs(t2) do
        if (torch.type(v) == "table") and (torch.type(t1[k] or false) == "table") then
            table.merge(t1[k], t2[k])
        else
            t1[k] = v
        end
    end
    return t1
end

--http://stackoverflow.com/questions/640642/how-do-you-copy-a-lua-table-by-value
function table.copy(t)
   if t == nil then
      return {}
   end
   local u = { }
   for k, v in pairs(t) do u[k] = v end
   return setmetatable(u, getmetatable(t))
end


function table.channelValue(tbl, channel, dept)
   dept = dept or 1
   if torch.type(tbl) ~= 'table' or dept > #channel then
      return tbl
   end
   return table.channelValue(tbl[channel[dept]], channel, dept+1)
end

function table.channelValues(tbls, channel)
   local values = {}
   for key, tbl in pairs(tbls) do
      values[key] = table.channelValue(tbl, channel)
   end
   return values
end

function table.fromString(str,splitter, f)
   f = f or  function(k,c) return tonumber(c) end
   if torch.type(str) == 'table' then
      return str
   end
   splitter = splitter or '[,]'
   return _.map(_.split(str:sub(2,-2),splitter), f)
end
