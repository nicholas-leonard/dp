------------------------------------------------------------------------
--[[ Postgres ]]--
-- Simplified PostgreSQL database connection handler. 
-- Uses the ~/.pgpass file for authentication 
-- see (http://wiki.postgresql.org/wiki/Pgpass)

-- TODO :
-- Cast row values to lua types using postgres types
------------------------------------------------------------------------
local Postgres = torch.class("dp.Postgres")
Postgres.isPostgres = true

function Postgres:__init(...)
   local args, database, user, host, env, autocommit
      = xlua.unpack(
      {... or {}},
      'Postgres', 'Default is to get the connection string from an ' ..
      'environment variable. For security reasons, and to allow ' .. 
      'for its serialization, no password is accepted. The password ' ..
      'should be set in the ~/.pgpass file.',
      {arg='database', type='string'},
      {arg='user', type='string'},
      {arg='host', type='string'},
      {arg='env', type='string', default='DEEP_PG_CONN'},
      {arg='autocommit', type='boolean', default=true}
   )
   if not (database or user or host) then
      self._conn_string = os.getenv(env)
   else
      error"NotImplementedError"
   end
   local env = require('luasql.postgres'):postgres()
   self._conn = assert(env:connect(self._conn_string))
   self._conn:setautocommit(autocommit)
   self._autocommit = autocommit
end
   
function Postgres:executeMany(command, param_list)
   local results = {}
   local result
   for i, params in pairs (param_list) do
      result = self:execute(command, params)
      table.insert(results, result)
   end
   return results
end
	
function Postgres:execute(command, params)
   local result
   if params then
      result = assert(
         self._conn:execute(
            string.format(
               command, unpack(params)
            )
         )
      )
   else
      result = self._conn:execute(command)
   end
   return result
end

--mode : 'n' returns rows as array, 'a' returns them as key-value
function Postgres:fetch(command, params, mode)
   mode = mode or 'n'
   local cur = self:execute(command, params)
   local coltypes = cur:getcoltypes()
   local colnames = cur:getcolnames()
   local row = cur:fetch({}, mode)
   local rows = {}
   while row do
     table.insert(rows, row)
     row = cur:fetch({}, mode)
   end
   cur:close()
   return rows, coltypes, colnames
end

function Postgres:fetchOne(command, params, mode)
   mode = mode or 'n'
   local cur = self:execute(command, params)
   local coltypes = cur:getcoltypes()
   local colname = cur:getcolnames()
   local row = cur:fetch({}, mode)
   cur:close()
   return row, coltypes, colnames
end

-- These two methods allow for (de)serialization of Postgres objects:
function Postgres:write(file)
   file:writeObject(self._conn_string)
   file:writeObject(self._autocommit)
end

function Postgres:read(file, version)
   self._conn_string = file:readObject()
   self._autocommit = file:readObject()
   local env = require('luasql.postgres'):postgres()
   self._conn = assert(env:connect(self._conn_string))
end


local function test(pg, no_serialize)
   local pg = pg or dp.Postgres()
   local res = pg:execute"CREATE TABLE public.test5464 ( n INT4, v FLOAT4, s TEXT )"
   print(res)
   res = pg:fetchOne"SELECT * FROM public.test5464"
   print(res)
   local param_list = {
      {5, 4.1, 'asdfasdf'},
      {6, 3.5, 'asdfashhd'},
      {6, 3.7, 'asdfashhd2'}
   }
   res = pg:executeMany("INSERT INTO public.test5464 VALUES (%s, %s, '%s');", param_list)
   print(res)
   res = pg:fetch"SELECT * FROM public.test5464 WHERE n = 6"
   print(res)
   if not no_serialize then
      local pg_str = torch.serialize(pg)
      test(torch.deserialize(pg_str), true)
   end
end


------------------------------------------------------------------------
--[[ PGEIDGenerator ]]--
-- EIDGenerator
-- Generates a unique identifier for the experiment using a postgresql
-- sequence.
------------------------------------------------------------------------

local PGEIDGenerator, parent 
   = torch.class("dp.PGEIDGenerator", "dp.EIDGenerator")
PGEIDGenerator.isPGEIDGenerator = true

function PGEIDGenerator:__init(config)
   local args, pg, sequence = xlua.unpack(
      {config or {}},
      'PGEIDGenerator', 'Generates unique IDs for experiments using ' ..
      'a postgreSQL Sequence.',
      {arg='pg', type='dp.Postgres', default=dp.Postgres()},
      {arg='sequence', type='string', default='dp.xp_id_gen'}
   )
   self._pg = pg
   self._sequence = sequence
end

function PGEIDGenerator:nextID()
   local eid = self._pg:fetchOne("SELECT nextval('%s')", 
                                 {self._sequence})[1]
   return dp.ObjectID(eid)
end


------------------------------------------------------------------------
--[[ PGLogger ]]--
-- Observer
-- Logger that logs a serialized report every epoch in a postgresql
-- database management system. Each epoch report is stored as a row in 
-- a table. Logger can only be used with PGEIDGenerator.

-- Note: if all sub-objects have a hierarchical unique id, they can be 
-- serialized
------------------------------------------------------------------------
local PGLogger, parent = torch.class("dp.PGLogger")
function PGLogger:__init(config)
   local args, pg = xlua.unpack(
      {config or {}},
      'PostgresLogger', nil,
      {arg='pg', type='dp.Postgres', default=dp.Postgres()}
   )
   parent.__init(self)
   self._pg = pg
end

--Actually this doesn't matter since report is sent by experiment
--[[function PGLogger:setSubject(subject)
   assert(subject.isExperiment)
   self._subject = subject
end]]--

function PGLogger:doneEpoch(report)
   local report_str = torch.serialize(report)
   self._pg:execute(
      "INSERT INTO dp.report (xp_id, epoch, report) " .. 
      "VALUES (%s, %s, '%s')", 
      {report.id, report.epoch, report_str}
   )
end

------------------------------------------------------------------------
--[[ PGQuery ]]--
-- Used to analyse patterns in the postgresql logs
------------------------------------------------------------------------
local PGQuery = torch.class("PGQuery")

function PGQuery:__init()
   
end

function PGQuery:design(design_id)
   
end

function PGQuery:reports(xp_id)

end

function PGQuery:report(xp_id, epoch)

end


