require 'dp'
async = require 'async'

-- tcp server
local server = async.tcp.listen({host='localhost', port=8483}, function(client)
   print('new connection:',client)
   client.ondata(function(data)
      data = torch.deserialize(data)
      print('received:',data)
      client.write(data)
   end)
   client.onend(function()
      print('client ended')
   end)
   client.onclose(function()
      print('closed.')
      collectgarbage()
      print(collectgarbage("count") * 1024)
   end)
end)

async.repl()



-- session (tcp client)
local wait = require 'async.fiber'.wait
local sync = require 'async.fiber'.sync
local exec = require 'async.process'.exec

fiber(function()
   -- wait on one function:
   local result,aux = wait(setTimeout, {1000}, function(timer)
      return 'something produced asynchronously', 'test'
   end)
   print(result,aux)

   -- wait on multiple functions:
   local results = wait({setTimeout, setTimeout}, {{500},{1000}}, function(timer)
      return 'some result',timer
   end)
   print(results)
   
   -- spawn job, default callback
   local res = wait(exec, {'ls', {'-l'}})
   print(res)

   -- we also provide a nicer syntax, that autowraps the most common calls (sync,exec,...):
   local res = sync.exec('ls', {'-al'})
   print(res)
end)

async.go()
