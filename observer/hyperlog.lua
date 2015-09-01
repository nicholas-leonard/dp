------------------------------------------------------------------------
--[[ HyperLog ]]--
-- Keeps a log of all reports and of the last minima
------------------------------------------------------------------------
local HyperLog, parent = torch.class("dp.HyperLog", "dp.Logger")

function HyperLog:__init()
   parent.__init(self, {"doneEpoch", "errorMinima"})
   self.reports = {}
   self.minimaEpoch = -1
end

function HyperLog:doneEpoch(report)
   if report.epoch > 0 then
      self.reports[report.epoch] = report
   end
end

function HyperLog:errorMinima(foundMinima, em)
   if foundMinima then
      self.minimaVal, self.minimaEpoch = em:minima()
   end
end

function HyperLog:getResultsByEpoch(filter, separator)
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local res = {}
   for k,v in pairs(filter) do
      assert(torch.type(k) == 'string')
      assert(torch.type(v) == 'string')
      local channel = _.split(v,separator)
      res[k] = table.channelValues(self.reports, channel)
   end
   return res
end

function HyperLog:getResultByEpoch(filter, separator)
   assert(torch.type(filter) == 'string')
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local channel = _.split(filter, separator)
   return table.channelValues(self.reports, channel)
end

function HyperLog:getResultsAtMinima(filter, separator)
   assert(torch.type(filter) == 'table')
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local report = self.reports[self.minimaEpoch]
   local res = {}
   for k,v in pairs(filter) do
      assert(torch.type(k) == 'string')
      assert(torch.type(v) == 'string')
      local channel = _.split(v,separator)
      res[k] = table.channelValue(report, channel)
   end
   return res
end

function HyperLog:getResultAtMinima(filter, separator)
   assert(torch.type(filter) == 'string')
   separator = separator or ':'
   assert(torch.type(separator) == 'string')
   local report = self.reports[self.minimaEpoch]
   local channel = _.split(filter, separator)
   return table.channelValue(report, channel)
end

