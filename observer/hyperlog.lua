------------------------------------------------------------------------
--[[ HyperLog ]]--
-- Keeps a log of all reports and of the last minima
------------------------------------------------------------------------

local HyperLog, parent = torch.class("dp.HyperLog", "dp.Logger")

function HyperLog:__init()
   parent.__init(self, {"doneExperiment", "errorMinima"})
   self.reports = {}
   self._minima_epoch = -1
end

function HyperLog:doneEpoch(report)
   if report > 0 then
      self.reports[report.epoch] = report
   end
end

function HyperLog:errorMinima(foundMinima, em)
   self._minima, self._minima_epoch = em:minima()
end

function HyperLog:getResultsByEpoch(filter, separator)
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local res = {}
   for k,v in pairs(filter) do
      assert(torch.type(k) == 'string')
      assert(torch.type(v) == 'string')
      local channel = _.split(v,separator)
      res[k] = dp.channelValues(self.reports, channel)
   end
   return res
end

function HyperLog:getResultByEpoch(filter, separator)
   assert(torch.type(filter) == 'string')
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local channel = _.split(filter, separator)
   return dp.channelValues(self.reports, channel)
end

function HyperLog:getResultsAtMinima(filter, separator)
   assert(torch.type(filter) == 'table')
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local report = self.reports[self._minima_epoch]
   local res = {}
   for k,v in pairs(filter) do
      assert(torch.type(k) == 'string')
      assert(torch.type(v) == 'string')
      local channel = _.split(v,separator)
      res[k] = dp.channelValue(report, channel)
   end
   return res
end

function HyperLog:getResultAtMinima(filter, separator)
   assert(torch.type(filter) == 'string')
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local report = self.reports[self._minima_epoch]
   local channel = _.split(filter, separator)
   return dp.channelValues(report, channel)
end

