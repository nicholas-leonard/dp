------------------------------------------------------------------------
--[[ SaveToFile ]]--
-- Strategy. Not an Observer.
-- Saves version of the subject with the lowest error
------------------------------------------------------------------------

local SaveToFile = torch.class("dp.SaveToFile")

function SaveToFile:__init(save_dir)
   self._save_dir = save_dir or dp.SAVE_DIR
end

function SaveToFile:setup(subject)
   --concatenate save directory with subject id
   self._filename = paths.concat(self._save_dir, 
                           subject:id():toPath() .. '.dat')
   --creates directories if required
   os.execute('mkdir -p ' .. sys.dirname(self._filename))
end

function SaveToFile:filename()
   return self._filename
end

function SaveToFile:save(subject)
   print('SaveToFile: saving to '.. self._filename)
   --save subject to file
   return torch.save(self._filename, subject)
end
