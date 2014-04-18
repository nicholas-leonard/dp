require 'torch'
async = require 'async'

local separator = '<347 SPLIT 879>'
local mode = 'binary'
local sent = 0
local recv = 0
local nClients = 10

local tensor = torch.randn(8,1100)

clients = {}

-- session (tcp client)
for i = 1,nClients do
   local client = async.tcp.connect('tcp://localhost:8483/', function(client)
      print('new connection:',client)
      client.onsplitdata(separator, function(data)
         print('received:', #data)
         local tensor = torch.deserialize(data, mode)
         print('received tensor :', tensor:size())
         recv = recv+1
         data = torch.serialize(tensor, mode)
         print('sending data', #data)
         client.write(data)
         client.write(separator)
         sent=sent+1
      end)
      client.onend(function()
         print('client ended')
      end)
      client.onclose(function()
         print('closed.')
      end)
      --client.write('test')

      
      local data = torch.serialize(tensor, mode)
      print('sending :', #data)
      client.write(data)
      client.write(separator)
      sent=sent+1

      async.setTimeout(2000, function()
         client.close()
      end)
   end)
   table.insert(clients, client)
end

local start_time = os.time()
async.go()
local period = os.time() - start_time
print("Sent "..sent.." tensors")
print("Recv "..recv.." tensors")
print("Speed")
print(sent/period.." tensor-send/second")
print(recv/period.."  tensor-recv/second")
print((recv+sent)/period.." tensors-transmissions/second")

