
------------------------------------------------------------------------
--[[ Logger ]]--
-- Observer
-- Simple logger that prints a report every epoch
------------------------------------------------------------------------

local Logger = torch.class("dp.Logger", "dp.Observer")

function Logger:doneEpoch(report)
   print(table.tostring(report))
end
