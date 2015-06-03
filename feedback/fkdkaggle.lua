------------------------------------------------------------------------
--[[ FKDKaggle ]]--
-- Feedback
-- Prepares a kaggle submission
-- Requires csvigo
------------------------------------------------------------------------
local FKDKaggle, parent = torch.class("dp.FKDKaggle", "dp.Feedback")
FKDKaggle.isFKDKaggle = true

FKDKaggle._submission_map = {
   ['left_eye_center_x'] = 1, ['left_eye_center_y'] = 2,
   ['right_eye_center_x'] = 3, ['right_eye_center_y'] = 4,
   ['left_eye_inner_corner_x'] = 5, ['left_eye_inner_corner_y'] = 6,
   ['left_eye_outer_corner_x'] = 7, ['left_eye_outer_corner_y'] = 8,
   ['right_eye_inner_corner_x'] = 9, ['right_eye_inner_corner_y'] = 10,
   ['right_eye_outer_corner_x'] = 11, ['right_eye_outer_corner_y'] = 12,
   ['left_eyebrow_inner_end_x'] = 13, ['left_eyebrow_inner_end_y'] = 14,
   ['left_eyebrow_outer_end_x'] = 15, ['left_eyebrow_outer_end_y'] = 16,
   ['right_eyebrow_inner_end_x'] = 17, ['right_eyebrow_inner_end_y'] = 18,
   ['right_eyebrow_outer_end_x'] = 19, ['right_eyebrow_outer_end_y'] = 20,
   ['nose_tip_x'] = 21, ['nose_tip_y'] = 22,
   ['mouth_left_corner_x'] = 23, ['mouth_left_corner_y'] = 24,
   ['mouth_right_corner_x'] = 25, ['mouth_right_corner_y'] = 26,
   ['mouth_center_top_lip_x'] = 27, ['mouth_center_top_lip_y'] = 28,
   ['mouth_center_bottom_lip_x'] = 29, ['mouth_center_bottom_lip_y'] = 30
}
   
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
   self._template = submission
   self._submission = {{submission[1][1],submission[1][4]}}
   self._file_name = file_name
   parent.__init(self, config)
   self._pixels = torch.range(0,97):float():view(1,1,98)
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._i = 2
   self._path = paths.concat(self._save_dir, self._file_name)
end

function FKDKaggle:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("errorMinima", self, "errorMinima")
end

function FKDKaggle:_add(batch, output, report)
   local target = batch:targets():forward('b')
   assert(output:dim() == 3)
   local act = output
   if torch.type(output) ~= torch.FloatTensor() then
      self._act = self._act or torch.FloatTensor()
      self._act:resize(output:size()):copy(output)
      act = self._act
   end
   local pixels = self._pixels:expandAs(act)
   self._output:cmul(act, pixels)
   self._keypoints:sum(self._output, 3)
   for i=1,act:size(1) do
      local keypoint = self._keypoints[i]:select(2,1)
      local row = self._template[self._i]
      local imageId = tonumber(row[2])
      assert(imageId == target[i])
      while (imageId == target[i]) do
         row = self._template[self._i]
         if not row then
            break
         end
         imageId = tonumber(row[2])
         local keypointName = row[3]
         self._submission[self._i] = {
            row[1], keypoint[self._submission_map[keypointName]]
         }
         self._i = self._i + 1
      end
   end
end

function FKDKaggle:_reset()
   self._i = 2
end

function FKDKaggle:errorMinima(found_minima)
   if found_minima then
      csvigo.save{path=self._path,data=self._submission,mode='raw'}
   end
end

