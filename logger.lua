
------------------------------------------------------------------------
--[[ Logger ]]--
-- Responsible for keeping track of the experiment as it progresses
-- for later analysis.
------------------------------------------------------------------------

local Logger = torch.class("dp.Logger")

function Logger:__init()
   self._experiment_log = {}
end

function Logger:logEpoch(epoch_log)
   self._epoch_log = epoch_log
   self._experiment_log[epoch_log:epoch()] = epoch_log
   print(table.tostring(epoch_log))
end

function Logger:channelValue(
