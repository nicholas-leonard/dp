require 'dp'

local xplog = dp.PGXpLog()
for i, row in ipairs(xplog:listCollections()) do
   print(i, row.collection_name, row.count, row.max)
end
