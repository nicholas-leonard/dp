require 'dp'

--[[parse command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('List experiments by descending minima')
cmd:text('Example:')
cmd:text('$> th rankbyminima.lua --collection "MNIST-NDT-2"')
cmd:text('Options:')
cmd:option('--collection', '', 'comma separated xplog collection id')
cmd:option('--limit', 10, 'number of top experiments to list')
cmd:option('--testChannel', 'tester:feedback:confusion:accuracy', 'report channel containing the test error')
cmd:text()
opt = cmd:parse(arg or {})


local xplog = dp.PGXpLog()
local rows = xplog:listMinima(opt.collection, opt.limit)
local entries = {}
local head = {valid={},test={},epoch={},xp_id={}}
for i, row in ipairs(rows) do
   head.valid[i] = tonumber(string.sub(row.minima,1,7))
   local epoch = tonumber(row.epoch)
   head.epoch[i] = epoch
   head.xp_id[i] = tonumber(row.xp_id)
   entry = xplog:entry(row.xp_id)
   head.test[i] = tonumber(table.channelValue(entry:report(epoch), _.split(opt.testChannel, ':')))
   table.insert(entries, entry)
end

local sheet = {}
for i, entry in ipairs(entries) do
   for k,v in pairs(entry:hyperReport().hyperparam) do
      local values = sheet[k] or {}
      if type(v) == 'table' then
         v = table.tostring(v)
      end
      values[i] = v
      sheet[k] = values
   end
end

print"------------RANK BY MINIMA----------------"
for k,v in pairs(head) do
   print(k, table.tostring(v))
end
for i,k in ipairs(_.sort(_.keys(sheet))) do
   if k ~= 'classes' then
      print(k, table.tostring(sheet[k]))
   end
end



