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
      {arg='submission', type='table', req=true},
      {arg='file_name', type='string', req=true},
      {arg='save_dir', type='string', help='defaults to dp.SAVE_DIR'},
      {arg='name', type='string', default='FKDKaggle'}
   )
   require 'csvigo'
   config.name = name
   self._save_dir = save_dir
   self._submission = submission
   self._file_name = file_name
   parent.__init(self, config)
   self._pixels = torch.randn(0,97):float()
   self._pixelView = torch.FloatTensor()
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._keypoint = torch.FloatTensor()
   self._i = 2
end

function FKDKaggle:setup(config)
   parent.setup(self, config)
   self._path = paths.concat(self._save_dir, self._file_name)
   self._mediator:subscribe("foundMinima", self, "foundMinima")
end

function FKDKaggle:_add(batch, output, carry, report)
   local act = output:forward('bwc', 'torch.FloatTensor')
   self._pixelView:view(self._pixels, act:size())
   self._output:cmul(act, self._pixelView)
   self._keypoints:sum(self._output, 3)
   for i=1,act:size(1) do
      local keypoint = self._keypoints:select(1,i):select(2,1)
      for j=1,act:size(2) do
         self._submision[self._i][4] = keypoint[j]
         self._i = self._i + 1
      end
   end
end

function FKDKaggle:_reset()
   self._i = 2
end

function FKDKaggle:foundMinima()
   csvigo.save{path=self._path,data=self._submission,mode='raw'}
end

