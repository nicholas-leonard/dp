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
   local client = async.tcp.connect({host='localhost', port='8080'}, function(client)
      async.fiber(function()
         client.sync()
         local data = torch.serialize(tensor, mode)
         print(i..' sending :', #data, sent)
         
         client.write(data)
         client.write(separator)
         print(i..' waiting for reply')
         local data = client.readsplit(separator)
         local tensor = torch.deserialize(data, mode)
         print(i..' setting result')
         result = tensor
         client.close()
         print(i, 'after close')
      end)
   end)
   table.insert(clients, client)
end

local start_time = os.time()
print('async go')
async.go()
local period = os.time() - start_time
print("Sent "..sent.." tensors")
print("Recv "..recv.." tensors")
print("Speed")
print(sent/period.." tensor-send/second")
print(recv/period.."  tensor-recv/second")
print((recv+sent)/period.." tensors-transmissions/second")

