------------------------------------------------------------------------
--[[ PGEarlyStopper ]]--
-- Observer, EarlyStopper
-- Maintains a postgreSQL table monitoring the epochs and errors of 
-- successive minima.
------------------------------------------------------------------------

local PGEarlyStopper, parent 
   = torch.class("dp.PGEarlyStopper", "dp.EarlyStopper")

function PGEarlyStopper:__init(config) 
   config = config or {}
   local args, pg, save_strategy = xlua.unpack(
      {config},
      'PGEarlyStopper', nil,
      {arg='pg', type='dp.Postgres', help='default is dp.Postgres()'},
      {arg='save_strategy', type='object', req=true,
       help='a serializable object that has a :save(subject) method.'}
   )
   config.save_strategy = save_strategy
   parent.__init(self, config)
   self._pg = pg or dp.Postgres()
end

function PGEarlyStopper:setSubject(subject)
   assert(subject.isExperiment)
   self._subject = subject
end

function PGEarlyStopper:setup(config)
   parent.setup(self, config)
   local channel_name = self._error_report or self._error_channel
   if type(channel_name) == 'table' then
      local channel_table = channel_name
      channel_name = ''
      for i, k in ipairs(channel_table) do
         channel_name = channel_name .. ':' .. k
      end
   end
   self._pg:execute(
      "INSERT INTO dp.earlystopper (xp_id, maximize, channel_name) " ..
      "VALUES (%s, %s, '%s')",
      {self._subject:name(), self._maximize, channel_name}
   )
end

function PGEarlyStopper:compareError(current_error, ...)
   if parent.compareError(self, current_error, ...) then
      self._pg:execute(
         "INSERT INTO dp.minima (xp_id, minima_epoch, minima_error) " ..
         "VALUES (%s, %s, %s)",
         {self._subject:name(), self._minima_epoch, self._minima}
      )
   end
end
