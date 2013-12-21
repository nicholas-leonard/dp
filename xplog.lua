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
