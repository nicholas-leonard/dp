------------------------------------------------------------------------
--[[ PGSaveToFile ]]--
-- Strategy. SaveToFile. 
-- Saves version of the subject with the lowest error to a file.
-- Maintains a postgreSQL table monitoring location of these files, 
-- which is useful when hyper-optimization is distributed on different
-- machines and file systems.
------------------------------------------------------------------------

local PGSaveToFile, parent = torch.class("dp.PGSaveToFile", "dp.SaveToFile")

function PGSaveToFile:__init(config)
   config = config or {}
   local args, pg, hostname = xlua.unpack(
      {config},
      'PGEarlyStopper', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'},
      {arg='hostname', type='string', default='localhost',
       help='hostname of this host'}
   )
   self._pg = pg or dp.Postgres()
   self._hostname = hostname
end

function PGSaveToFile:setup(subject)
   parent.setup(self, subject)
   self._pg = pg
   self._pg:execute(
      "INSERT INTO dp.savetofile (xp_id, hostname, filename) " ..
      "VALUES (%s, '%s', '%s');",
      {subject:id(), self._hostname, self._filename}
   )
end

