
------------------------------------------------------------------------
--[[ Logger ]]--
-- Observer
-- Simple logger that prints a report every epoch
-- TODO use TREPL to make nice reports
------------------------------------------------------------------------

local Logger, parent = torch.class("dp.Logger", "dp.Observer")
Logger.isLogger = true

function Logger:__init()
   parent.__init(self, "doneEpoch")
end

function Logger:doneEpoch(report)
   --print(table.tostring(report))
end
