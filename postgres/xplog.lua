------------------------------------------------------------------------
--[[ PGXpLog ]]--
-- Composite of PGXpLogEntries
------------------------------------------------------------------------
local PGXpLog, parent = torch.class("dp.PGXpLog", "dp.XpLog")
PGXpLog.isPGXpLog = true

function PGXpLog:__init(config)
   config = config or {}
   local args, pg = xlua.unpack(
      {config},
      'PGLogQuery', nil,
      {arg='pg', type='dp.Postgres', help='defaults to dp.Postgres()'}
   )
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGXpLog:createCollection(collection_name)
   assert(type(collection_name) == 'string')
   -- just make sure the collection exists :
   local rows = self._pg:fetch(
      "SELECT xp_id " .. 
      "FROM dp.experiment " ..
      "WHERE collection_name = '%s'", 
      {collection_name},
      'a'
   )
   if _.isEmpty(rows) then
      return
   end
   local collection = {}
   for i, row in ipairs(rows) do
      table.insert(collection, self:entry(tonumber(row.xp_id)))
   end 
   return collection
end

function PGXpLog:createEntry(xp_id)
   return dp.PGXpLogEntry{id=xp_id, pg=self._pg}
end

function PGXpLog:listCollections()
   local rows = self._pg:fetch(
      "SELECT collection_name, COUNT(*), MAX(start_time) " .. 
      "FROM dp.experiment " ..
      "GROUP BY collection_name ORDER BY max DESC",
      {}, 'a'
   )
   if not _.isEmpty(rows) then 
      return rows
   end
end

function PGXpLog:listMinima(collection_names, limit)
   if type(collection_names) == 'string' then
      collection_names = _.split(collection_names, separator or ',')
   end
   sql_cln_names = ""
   for i, cln_name in ipairs(collection_names) do
      sql_cln_names = sql_cln_names .. "'" .. cln_name .. "',"
   end
   local rows = self._pg:fetch(
      "SELECT a.xp_id, MIN(minima_error) AS minima, "..
         "MAX(minima_epoch) AS epoch "..
      "FROM dp.experiment AS a, dp.minima AS b "..
      "WHERE collection_name IN (%s) AND a.xp_id = b.xp_id "..
      "GROUP BY a.xp_id ORDER BY minima ASC LIMIT %s",
      {string.sub(sql_cln_names, 1, -2), limit or 100}, 'a'
   )
   if not _.isEmpty(rows) then 
      return rows
   end
end

------------------------------------------------------------------------
--[[ PGXpLogEntry ]]--
-- XpLogEntry
-- maps to the database entity-relations representing experiments
------------------------------------------------------------------------
local PGXpLogEntry, parent = torch.class("dp.PGXpLogEntry", "dp.XpLogEntry")

function PGXpLogEntry:__init(config)
   config = config or {}
   local args, pg = xlua.unpack(
      {config},
      'PGXpLogEntry', nil,
      {arg='pg', type='dp.Postgres', help='defaults to dp.Postgres()'}
   )
   if type(config.id) == 'string' then
      config.id = tonumber(config.id)
   end
   parent.__init(self, config)
   self._pg = pg
   local row = self:selectExperiment(self._id)
   if not row then 
      error"PGXpLogEntry : not data in database"
   end
   self._collection_name = row.collection_name
   self._process_name = row.process_name
   self._hyper_report = torch.deserialize(row.hyper_report_pickle, 'ascii')
   self._start_time = start_time
   self._reports = {}
end

function PGXpLogEntry:refreshDone()
   local row = self:selectDone(self._id)
   self._dirty = true
   if row then
      self._end_time = row.end_time
      self._dirty = false
   end 
end

function PGXpLogEntry:refresh()
   self:refreshDone()
   self:refreshReports()
end

function PGXpLogEntry:report(report_epoch)
   local report = self._reports[report_epoch]
   if (not report) then
      local row = self:selectReport(self._id, report_epoch)
      return torch.deserialize(row.report_pickle, 'ascii')
   end
   return report
end

function PGXpLogEntry:refreshReports()
   local rows = self:selectReports(self._id, #self._reports)
   if rows then
      for i, row in ipairs(rows) do
         local report = torch.deserialize(row.report_pickle, 'ascii')
         table.insert(self._reports, tonumber(row.report_epoch), report)
      end
   end
end

function PGXpLogEntry:minima()
   local rows = self:selectMinima(self._id)
   local epochs = {}
   local minimas = {}
   if rows then
      for i, row in ipairs(rows) do
         table.insert(epochs, tonumber(row.minima_epoch))
         table.insert(minimas, tonumber(row.minima_error))
      end
   end
   local row = self:selectEarlyStopper(self._id)
   local sign = 1
   if row then
      if row.maximize == 't' then
         sign = -1
      end
   end
   epochs = torch.Tensor(epochs)
   minimas = torch.Tensor(minimas)
   return epochs, minimas:mul(sign), epochs[-1], minimas[-1]
end

function PGXpLogEntry:plotLearningCurves(...)
   local curves, x_name, y_name, x = self:_plotReportChannel(...)
   local epochs, minimas, epoch, minima = self:minima()
   table.insert(curves, {'minima', epochs, minimas, '+'})
   gnuplot.title('Learning Curves. minima : '..minima..'@'..epoch)
   if x:nElement() > 1 then
      gnuplot.plot(unpack(curves))
   end
end
   

function PGXpLogEntry:selectExperiment(xp_id)
   assert(type(xp_id) == 'number')
   local row = self._pg:fetchOne(
      "SELECT collection_name, process_name, " ..
      "       hyper_report_pickle, start_time " .. 
      "FROM dp.experiment " ..
      "WHERE xp_id = %s", 
      {xp_id}, 'a'
   )
   if not _.isEmpty(row) then
      return row
   end
end

function PGXpLogEntry:selectDone(xp_id)
   assert(type(xp_id) == 'number')
   local row = self._pg:fetchOne(
      "SELECT end_time " .. 
      "FROM dp.done " ..
      "WHERE xp_id = %s", 
      {xp_id}, 'a'
   )
   if not _.isEmpty(row) then
      return row
   end
end

function PGXpLogEntry:selectReport(xp_id, epoch)
   assert(type(xp_id) == 'number' and type(epoch) == 'number')
   local row = self._pg:fetchOne(
      "SELECT report_pickle " .. 
      "FROM dp.report " ..
      "WHERE (xp_id, report_epoch) = (%s, %s)", 
      {xp_id, epoch}, 'a'
   )
   if not _.isEmpty(row) then
      return row
   end
end

function PGXpLogEntry:selectReports(xp_id, min_epoch)
   min_epoch = min_epoch or 0
   assert(type(xp_id) == 'number')
   local rows = self._pg:fetch(
      "SELECT report_epoch, report_pickle " .. 
      "FROM dp.report " ..
      "WHERE xp_id = %s AND report_epoch >= %s", 
      {xp_id, min_epoch}, 'a'
   )
   if not _.isEmpty(rows) then 
      return rows
   end
end

function PGXpLogEntry:selectEarlyStopper(xp_id)
   assert(type(xp_id) == 'number')
   local row = self._pg:fetchOne(
      "SELECT maximize, channel_name " .. 
      "FROM dp.earlystopper " ..
      "WHERE xp_id = %s", 
      {xp_id}, 'a'
   )
   if not _.isEmpty(row) then 
      return row
   end
end

function PGXpLogEntry:selectMinima(xp_id, min_epoch)
   assert(type(xp_id) == 'number')
   min_epoch = min_epoch or 0
   local rows = self._pg:fetch(
      "SELECT minima_epoch, minima_error " .. 
      "FROM dp.minima " ..
      "WHERE xp_id = %s AND minima_epoch >= %s" , 
      {xp_id, min_epoch}, 'a'
   )
   if not _.isEmpty(rows) then 
      return rows
   end
end

local function xplogtest()
   local pg = dp.Postgres()
   local xplog = dp.PGXpLog{pg=pg}
   local entry = xplog:entry(992)
   local cln = xplog:collection('MnistMLP1')
   print(entry:reports())
   print(entry:hyperReport())
   print(#cln, cln[1])
end
