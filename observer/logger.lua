------------------------------------------------------------------------
--[[ Logger ]]--
-- Interface, Observer
-- Simple logger that prints a report every epoch
------------------------------------------------------------------------

local Logger, parent = torch.class("dp.Logger", "dp.Observer")
Logger.isLogger = true

function Logger:__init(channels)
   parent.__init(self, channels or {"doneExperiment", "doneEpoch"})
end

function Logger:doneEpoch(report)
   print(report)
end

function Logger:doneExperiment(report)

end

function Logger:logHyperReport(hyper_report)
   print(hyper_report)
end
