------------------------------------------------------------------------
--[[ SaveToFile ]]--
-- Strategy. Not an Observer.
-- Saves version of the subject with the lowest error
------------------------------------------------------------------------
local SaveToFile = torch.class("dp.SaveToFile")
SaveToFile.isSaveToFile = true

function SaveToFile:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args, in_memory, save_dir, verbose = xlua.unpack(
      {config},
      'SaveToFile', 
      'Saves version of the subject with the lowest error',
      {arg='in_memory', type='boolean', default=false, 
       help='only saves the subject to file at the end of the experiment'},
      {arg='save_dir', type='string', help='defaults to dp.SAVE_DIR'},
      {arg='verbose', type='boolean', default=true,
       help='can print messages to stdout'}
   )
   self._in_memory = in_memory
   self._save_dir = save_dir or dp.SAVE_DIR
   self._verbose = verbose
end

function SaveToFile:setup(subject, mediator)
   self._mediator = mediator
   if self._in_memory then
      self._mediator:subscribe('doneExperiment', self, 'doneExperiment')
   end
   
   --concatenate save directory with subject id
   self._filename = paths.concat(self._save_dir, subject:id():toPath() .. '.dat')
   os.execute('mkdir -p ' .. sys.dirname(self._filename))
end

function SaveToFile:filename()
   return self._filename
end

function SaveToFile:save(subject)
   assert(subject, "SaveToFile not setup error")
   if self._in_memory then
      dp.vprint(self._verbose, 'SaveToFile: serializing subject to memory')
      self._save_cache = nil
      self._save_cache = torch.serialize(subject)
   else
      dp.vprint(self._verbose, 'SaveToFile: saving to '.. self._filename)
      return torch.save(self._filename, subject)
   end
end

function SaveToFile:doneExperiment()
   if self._in_memory and self._save_cache then
      dp.vprint(self._verbose, 'SaveToFile: saving to '.. self._filename)
      local f = io.open(self._filename, 'w')
      f:write(self._save_cache)
      f:close()
   end
end

-- the following are called by torch.File during [un]serialization
function SaveToFile:write(file)
   -- prevent subject from being serialized twice
   local state = _.map(self, 
      function(k,v) 
         if k ~= '_save_cache' then 
            return v;
         end
      end)
   
   file:writeObject(state)
end

function SaveToFile:read(file)
   local state = file:readObject()
   for k,v in pairs(state) do
      self[k] = v
   end
end

function SaveToFile:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function SaveToFile:silent()
   self:verbose(false)
end
