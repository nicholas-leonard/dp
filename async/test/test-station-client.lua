async = require 'async'
require 'torch'

local mode = 'ascii'
local separator = '<347 SPLIT 879>'
local tensor = torch.randn(8,1100)

local client = async.tcp.connect({host='localhost', port='8080'}, function(client)
   print('new connection:',client)
   client.onsplitdata(separator, function(data)
      print('received:', #data)
      local tensor = torch.deserialize(data, mode)
      print('received tensor :', torch.typename(tensor))
      client.close()
   end)
   client.onend(function()
      print('client ended')
   end)
   client.onclose(function()
      print('closed.')
   end)
   local data = torch.serialize(tensor, mode)
   print('sending :', #data)
   client.write(data)
   client.write(separator)
   print('wrote')
end)

async.go()
