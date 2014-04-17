require 'torch'
async = require 'async'

local separator = '<347 SPLIT 879>'

local tensor = torch.randperm(400000):resize(400,1000)
local data = torch.serialize(tensor, 'ascii')

-- session (tcp client)
local client = async.tcp.connect('tcp://localhost:8483/', function(client)
   print('new connection:',client)
   client.onsplitdata(separator, function(data)
      print('received:', #data)
      client.write(data)
      client.write(separator)
   end)
   client.onend(function()
      print('client ended')
   end)
   client.onclose(function()
      print('closed.')
   end)
   --client.write('test')

   print('sending :', #data)
   client.write(data)
   client.write(separator)

   async.setTimeout(10000, function()
      client.close()
   end)
end)

async.go()
