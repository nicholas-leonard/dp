------------------------------------------------------------------------
--[[ Logger ]]--
-- Interface, Observer
-- Simple logger that prints a report every epoch
-- TODO use TREPL to make nice reports
------------------------------------------------------------------------

local Logger, parent = torch.class("dp.Logger", "dp.Observer")
Logger.isLogger = true

function Logger:__init()
   parent.__init(self, {"doneExperiment", "doneEpoch"})
end

function Logger:doneEpoch(report)
   print(report)
end

function Logger:doneExperiment(report)

end

function Logger:logHyperReport(hyper_report)
   print(hyper_report)
end
