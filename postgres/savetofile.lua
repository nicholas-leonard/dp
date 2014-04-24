------------------------------------------------------------------------
--[[ PGSaveToFile ]]--
-- Strategy. SaveToFile. 
-- Saves version of the subject with the lowest error to a file.
-- Maintains a postgreSQL table monitoring location of these files, 
-- which is useful when hyper-optimization is distributed on different
-- machines and file systems.
------------------------------------------------------------------------
local PGSaveToFile, parent = torch.class("dp.PGSaveToFile", "dp.SaveToFile")
PGSaveToFile.isPGSaveToFile = true

function PGSaveToFile:__init(config)
   config = config or {}
   local args, pg = xlua.unpack(
      {config},
      'PGEarlyStopper', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'}
   )
   self._pg = pg or dp.Postgres()
   self._hostname = os.hostname()
   parent.__init(self, config)
end

function PGSaveToFile:setup(subject)
   parent.setup(self, subject)
   self._pg:execute(
      "INSERT INTO dp.savetofile (xp_id, hostname, filename) " ..
      "VALUES (%s, '%s', '%s');",
      {subject:name(), self._hostname, self._filename}
   )
end

