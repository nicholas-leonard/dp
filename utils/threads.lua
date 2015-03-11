
function dp.Threads(...)
   local args = {...}
    -- https://github.com/torch/threads-ffi
	local Threads = require "threads"
   
   local ffi = require 'ffi'
   local sdl = require 'sdl2'
   local Worker = require 'threads.worker'
   local C = ffi.C
   local serialize = require 'threads.serialize'
   
   local LUA_GLOBALSINDEX = -10002;

   local function checkL(L, status)
      if not status then
         local msg = ffi.string(C.lua_tolstring(L, -1, NULL))
         error(msg)
      end
   end
   
   -- I tried to do a PR for these small changes, but was denied : 
   -- https://github.com/torch/threads-ffi/pull/9
   -- so here they are in my code.
   
	-- allow for more flexible contructor (we can specify size of queues)
	function Threads:__call(N, ...)
	   local self = {N=N, endcallbacks={n=0}, errors={}}
	   local funcs = {...}
	   
	   local M = N -- size of queues
	   if torch.type(funcs[1]) == 'number' then
		  M = table.remove(funcs,1)
	   end
	   
	   if #funcs == 0 then
		  funcs = {function() end}
	   end
	   
	   local initres = {}

	   setmetatable(self, {__index=Threads})

	   self.mainworker = Worker(M)
	   self.threadworker = Worker(M)

	   self.threads = {}
	   for i=1,N do
		  local L = C.luaL_newstate()
		  assert(L ~= nil, string.format('%d-th lua state creation failed', i))
		  C.luaL_openlibs(L)

		  for j=1,#funcs do
			 local code_p, sz = serialize.save(funcs[j])
			 if j < #funcs then
				checkL(L, C.luaL_loadstring(L, string.format([[
				  local serialize = require 'threads.serialize'
				  local ffi = require 'ffi'
				  local code = serialize.load(ffi.cast('const char*', %d), %d)
				  code(%d)
				]], tonumber(ffi.cast('intptr_t', code_p)), sz, i)))
			 else
				checkL(L, C.luaL_loadstring(L, string.format([[
				  local serialize = require 'threads.serialize'
				  local ffi = require 'ffi'
				  local code = serialize.load(ffi.cast('const char*', %d), %d)
				  __threadid = %d
				  __workerinitres_p, __workerinitres_sz = serialize.save{code(%d)}
				  __workerinitres_p = tonumber(ffi.cast('intptr_t', __workerinitres_p))
				]], tonumber(ffi.cast('intptr_t', code_p)), sz, i, i)))
			 end
			 checkL(L, C.lua_pcall(L, 0, 0, 0) == 0)
		  end

		  C.lua_getfield(L, LUA_GLOBALSINDEX, '__workerinitres_p')
		  local workerinitres_p = C.lua_tointeger(L, -1)
		  C.lua_getfield(L, LUA_GLOBALSINDEX, '__workerinitres_sz')
		  local workerinitres_sz = C.lua_tointeger(L, -1)
		  C.lua_settop(L, -3)
		  table.insert(initres, serialize.load(ffi.cast('const char*', workerinitres_p), workerinitres_sz))

		  checkL(L, C.luaL_loadstring(L, [[
	  local ffi = require 'ffi'
	  local sdl = require 'sdl2'
	  require 'threads.worker'

	  local function workerloop(data)
		 local workers = ffi.cast('struct THWorker**', data)
		 local mainworker = workers[0]
		 local threadworker = workers[1]
		 local threadid = __threadid

		 while __worker_running do
			local status, res, endcallbackid = threadworker:dojob()
			mainworker:addjob(function()
								 return status, res, endcallbackid, threadid
							  end)
		 end

		 return 0
	  end

	  __worker_running = true
	  __workerloop_ptr = tonumber(ffi.cast('intptr_t', ffi.cast('int (*)(void *)', workerloop)))
	]]
	) == 0)
		  checkL(L, C.lua_pcall(L, 0, 0, 0) == 0)
		  C.lua_getfield(L, LUA_GLOBALSINDEX, '__workerloop_ptr')
		  local workerloop_ptr = C.lua_tointeger(L, -1)
		  C.lua_settop(L, -2);

		  local workers = ffi.new('struct THWorker*[2]', {self.mainworker, self.threadworker}) -- note: GCed
		  local thread = sdl.createThread(ffi.cast('SDL_ThreadFunction', workerloop_ptr), string.format("%s%.2d", Threads.name, i), workers)
		  assert(thread ~= nil, string.format('%d-th thread creation failed', i))
		  table.insert(self.threads, {thread=thread, L=L})
	   end
      print"created"
	   return self, initres
	end
   
   -- threads-ffi changed this in such a way that it now crashes.
   function Threads:addjob(callback, endcallback, ...) -- endcallback is passed with returned values of callback
      if #self.errors > 0 then self:synchronize() end -- if errors exist, sync immediately.
      local endcallbacks = self.endcallbacks

      -- now add a new endcallback in the list
      local endcallbackid = table.getn(endcallbacks)+1
      endcallbacks[endcallbackid] = endcallback or function() end
      endcallbacks.n = endcallbacks.n + 1

      local func = function(...)
         local res = {pcall(callback, ...)}
         local status = table.remove(res, 1)
         return status, res, endcallbackid
      end

      self.threadworker:addjob(func, ...)
   end

	-- tell you if the queues are empty.
	function Threads:isEmpty()
	   return not (self.mainworker.runningjobs > 0 or self.threadworker.runningjobs > 0 or self.endcallbacks.n > 0)
	end

	return Threads(unpack(args))
end
