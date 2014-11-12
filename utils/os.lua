
--http://stackoverflow.com/questions/132397/get-back-the-output-of-os-execute-in-lua
function os.capture(cmd, raw)
   local f = assert(io.popen(cmd, 'r'))
   local s = assert(f:read('*a'))
   f:close()
   if raw then return s end
   s = string.gsub(s, '^%s+', '')
   s = string.gsub(s, '%s+$', '')
   s = string.gsub(s, '[\n\r]+', ' ')
   return s
end

function os.pid()
   return tonumber(_.split(os.capture('cat /proc/self/stat'), ' ')[5])
end

function os.hostname()
   local f = io.popen ("/bin/hostname")
   if not f then 
      return 'localhost'
   end
   local hostname = f:read("*a") or ""
   f:close()
   hostname =string.gsub(hostname, "\n$", "")
   return hostname
end
