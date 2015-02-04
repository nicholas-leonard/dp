require 'dp'
cjson = require 'cjson' --sudo luarocks install lua-cjson

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Load json string from file, decode into table and torch.save it to disk')
cmd:text('Options:')
cmd:option('--srcPath', 'json.txt', 'path to file containing json')
cmd:option('--dstPath', 'json.th7', 'path to file where serialized table will be saved')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)

file = io.open(opt.srcPath, 'r')
jsonString = file:read()
file:close()

tbl = cjson.decode(jsonString)

torch.save(opt.dstPath, tbl)
