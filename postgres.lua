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
   local args, database, user, host, password, env, autocommit
      = xlua.unpack(
      {... or {}},
      'Postgres', 'Default is to get the connection string from an ' ..
      'environment variable.',
      {arg='database', type='string'},
      {arg='user', type='string'},
      {arg='host', type='string'},
      {arg='password', type='string'},
      {arg='env', type='string', default='DEEP_PG_CONN'},
      {arg='autocommit', type='boolean', default=true}
   )
   local conn_string
   if not (database or user or host or password) then
      conn_string = os.getenv(env)
   else
      error"NotImplementedError"
   end
   local env = require('luasql.postgres'):postgres()
   self._conn = assert(env:connect(conn_string))
   self._conn:setautocommit(autocommit)
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

local function test()
   local pg = dp.Postgres()
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
end
