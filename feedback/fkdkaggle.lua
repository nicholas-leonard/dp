------------------------------------------------------------------------
--[[ FKDKaggle ]]--
-- Feedback
-- Prepares a kaggle submission
-- Requires csvigo
------------------------------------------------------------------------
local FKDKaggle, parent = torch.class("dp.FKDKaggle", "dp.Feedback")
FKDKaggle.isFKDKaggle = true
   
function FKDKaggle:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, submission, file_name, save_dir, name = xlua.unpack(
      {config},
      'FKDKaggle', 
      'Used to prepare a Kaggle Submission for the '..
      'Facial Keypoints Detection challenge',
      {arg='submission', type='table', req=true, 
       help='sample submission table'},
      {arg='file_name', type='string', req=true,
       help='name of file to save submission to'},
      {arg='save_dir', type='string', default=dp.SAVE_DIR,
       help='defaults to dp.SAVE_DIR'},
      {arg='name', type='string', default='FKDKaggle',
       help='name identifying Feedback in reports'}
   )
   require 'csvigo'
   config.name = name
   self._save_dir = save_dir
   self._submission = submission
   self._file_name = file_name
   parent.__init(self, config)
   self._pixels = torch.range(0,97):float():view(1,1,98)
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._keypoint = torch.FloatTensor()
   self._i = 2
   self._path = paths.concat(self._save_dir, self._file_name)
end

function FKDKaggle:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("foundMinima", self, "foundMinima")
end

function FKDKaggle:_add(batch, output, carry, report)
   local act = output:forward('bwc', 'torch.FloatTensor')
   local pixels = self._pixels:expandAs(act)
   self._output:cmul(act, pixels)
   self._keypoints:sum(self._output, 3)
   for i=1,act:size(1) do
      local keypoint = self._keypoints[i]:select(2,1)
      for j=1,act:size(2) do
         self._submission[self._i][4] = keypoint[j]
         self._i = self._i + 1
      end
   end
end

function FKDKaggle:_reset()
   self._i = 2
end

function FKDKaggle:foundMinima()
   print("FKDKaggle", self._i, #self._submission)
   csvigo.save{path=self._path,data=self._submission,mode='raw'}
end

