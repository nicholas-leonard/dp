------------------------------------------------------------------------
--[[ XpLog ]]--
-- Interface, Composite of CollectionQueries
------------------------------------------------------------------------
local XpLog = torch.class("dp.XpLog")
XpLog.isXpLog = true

function XpLog:__init(...)
   local args, name = xlua.unpack(
      {... or {}},
      'XpLog', nil,
      {arg='name', type='string | number'}
   )
   self._entries = {}
   self._collections = {}
end

function XpLog:collection(collection_name)
   local collection = self._collections[collection_name] 
   if not collection then 
      collection = self:createCollection(collection_name)
      self._collections[collection_name] = collection
   end
   if not collection then
      print"Query:collection() Warning : collection not found"
   end
   return collection
end

function XpLog:collections(collection_names, separator)
   if type(collection_names) == 'string' then
      collection_names = _.split(collection_names, separator or ',')
   end
   return _.map(
      collection_names, 
      function(c) return self:collection(c) end
   )
end

function XpLog:entry(xp_id)
   local entry = self._entries[xp_id]
   if not entry then
      entry = self:createEntry(xp_id)
      self._entries[xp_id] = entry
   end
   return entry
end

function XpLog:createEntry(xp_id)

end

------------------------------------------------------------------------
--[[ XpLogEntry ]]--
------------------------------------------------------------------------
local XpLogEntry = torch.class("dp.XpLogEntry")
XpLogEntry.isXpLogEntry = true

function XpLogEntry:__init(...)
   local args, id = xlua.unpack(
      {... or {}},
      'XpLogEntry', nil,
      {arg='id', type='string | number', req=true}
   )
   self._id = id
   self._reports = {}
   self._dirty = true
end

function XpLogEntry:report(report_epoch)
   local report = self._reports[report_epoch]
   if (not report) then
      self:sync()
      return self._reports[report_epoch]
   end
   return report
end

function XpLogEntry:sync(...)
   if self:dirty() then
      self:refresh(...)
      collectgarbage() 
   end
end

function XpLogEntry:refresh(...)

end

function XpLogEntry:dirty()
   return self._dirty
end

function XpLogEntry:reports()
   self:sync()
   return self._reports
end

function XpLogEntry:hyperReport()
   return self._hyper_report
end

function XpLogEntry:reportChannel(channel, reports)
   reports = reports or self:reports()
   if type(channel) == 'string' then
      channel = _.split(channel, ':')
   end
   return table.channelValues(reports, channel), reports
end

function XpLogEntry:_plotReportChannel(...)
   local args, channels, curve_names, x_name, y_name, hloc, vloc
      = xlua.unpack(
      {... or {}},
      'XpLogEntry:plotReportChannel', nil,
      {arg='channels', type='string | table', req=true},
      {arg='curve_names', type='string | table'},
      {arg='x_name', type='string', default='epoch'},
      {arg='y_name', type='string'},
      {arg='hloc', type='string', default='right'},
      {arg='vloc', type='string', default='bottom'}
   )
   if type(channels) == 'string' then
      channels = _.split(channels, ',') 
   end
   if type(curve_names) == 'string' then
      curve_names = _.split(curve_names, ',')
   end
   curve_names = curve_names or channels
   local reports = self:reports()
   local x = torch.Tensor(_.keys(reports))
   local curves = {}
   for i,channel in ipairs(channels) do
      local values = self:reportChannel(channel, reports)
      table.insert(
         curves, { curve_names[i], x, torch.Tensor(values), '-' }
      )
   end
   require 'gnuplot'
   --gnuplot.xlabel(x_name)
   --gnuplot.ylabel(y_name)
   --gnuplot.movelegend(hloc,vloc)
   return curves, x_name, y_name, x
end

function XpLogEntry:plotReportChannel(...)
   local curves, x_name, y_name, x = self:_plotReportChannel(...)
   if x:nElement() > 1 then
      gnuplot.plot(unpack(curves))
   end
end
