------------------------------------------------------------------------
--[[ TopCrop ]]--
-- Feedback for use with ImageNet-like dataset.
------------------------------------------------------------------------
local TopCrop, parent = torch.class("dp.TopCrop", "dp.Feedback")
   
function TopCrop:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, n_top, n_crop, center, name = xlua.unpack(
      {config},
      'TopCrop', 
      'Measures top-n classification accuracy for m-crops taken from '..
      'the same image (which therefore have the same target class).',
      {arg='n_top', type='number | table', 
       help='the accuracy is measured by looking for the target class'..
       ' in the n_top predictions with highest log-likelihood. '..
       ' Defaults to {1,5} ,i.e. top-1 and top-5'},
      {arg='n_crop', type='number', default=10,
       help='The number of crops taken from each sample. The assumption'..
       ' is that the input performs n_crop propagations of each image'..
       ' such that their predictions can be averaged'},
      {arg='center', type='number', default=2,
       help='The number of first crops to be considered center patches for '..
       'which performance will also be reported separately. '..
       'This means that you should put center crops first.'},
      {arg='name', type='string', default='topcrop',
       help='name identifying Feedback in reports'}
   )
   require 'torchx'
   config.name = name
   self._n_top = n_top or {1,5}
   self._n_top = (torch.type(n_top) == 'number') and {self._n_top} or self._n_top
   _.sort(self._n_top)
   self._n_crop = n_crop
   assert(center >= 1)
   self._center = center 
   parent.__init(self, config)
   self:reset()
end

function TopCrop:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function TopCrop:doneEpoch(report)
   if report.epoch > 0 then
      local msg = self._id:toString().." accuracy top ("..table.concat(self._n_top,",")..") :"
      local centerTops = {}
      local allTops = {}
      local nSample = self._n_sample
      for i,top in ipairs(self._n_top) do
         table.insert(centerTops, string.format("%g", self.topCounts.center[top]/nSample*100))
         table.insert(allTops, string.format("%g", self.topCounts.all[top]/nSample*100))
      end
      if self._verbose then
         msg = msg .. "center ("..table.concat(centerTops, ",").."); "
         msg = msg .. "all ("..table.concat(allTops,",")..")"
         print(msg)
      end
   end
end

function TopCrop:topstats(preds, targets, ns)
   -- sort each sample's prediction in descending order of activation
   self._sorted = self._sorted or preds.new()
   self._indices = self._indices or torch.LongTensor()
   self._sorted:sort(self._indices, preds, 2, true)
   
   local topClasses = self._indices:narrow(2,1,self._n_top[#self._n_top])
   
   for i=1,preds:size(1) do
      local p,g = topClasses[i], targets[i]
      -- torch.find is from torchx (in case you are wondering)
      local idx = torch.find(p,g)[1]
      for top, count in pairs(self.topCounts[ns]) do
         if idx and idx <= top then
            self.topCounts[ns][top] = count + 1 
         end
      end
   end
end

function TopCrop:add(batch, output, report)
   assert(output:dim() == 2)
   local preds = output
   if torch.type(output) ~= torch.FloatTensor() then
      self._act = self._act or torch.FloatTensor()
      self._act:resize(output:size()):copy(output)
      preds = self._act
   end
   local labels = batch:targets():forward('b')
   assert(preds:isContiguous())
   assert(labels:isContiguous())
   
   if math.fmod(preds:size(1), self._n_crop) ~= 0 then
      error("TopCrop: the number of samples should be a multiple of n_crop")
   end
   
   local predView = preds:view(preds:size(1)/self._n_crop, self._n_crop, preds:size(2))
   local labelView = labels:view(labels:size(1)/self._n_crop, self._n_crop)
   
   self._n_sample = self._n_sample + predView:size(1)
   
   -- check that each images n_crops have the same label
   self._labels = self._labels or torch.FloatTensor()
   self._labels:resize(labelView:size()):copy(labelView)
   self._labelStd = self._labelStd or torch.FloatTensor()
   if self._labelStd:std(self._labels, 2):max() ~= 0 then
      print(labelView)
      error"TopCrop: The n_crops per image should all have the same target label"
   end
   
   local targets = labelView:select(2,1)
   
   -- center(s)
   local center = predView:narrow(2,1,self._center)
   self._sum = self._sum or preds.new()
   self._sum:sum(center, 2)
   self:topstats(self._sum:select(2,1), targets, 'center')
   -- all crops
   self._sum:sum(predView, 2)
   self:topstats(self._sum:select(2,1), targets, 'all')
end

function TopCrop:_reset()
   self.topCounts = {all={}, center={}}
   for j,top in ipairs(self._n_top) do
      self.topCounts.all[top] = 0
      self.topCounts.center[top] = 0
   end
end

function TopCrop:report()
   local centerTops = {}
   local allTops = {}
   local nSample = self._n_sample
   for i,top in ipairs(self._n_top) do
      centerTops[top] = self.topCounts.center[top]/nSample*100
      allTops[top] = self.topCounts.all[top]/nSample*100
   end
   return { 
      [self:name()] = {
         center=centerTops, all=allTops, n_crop=self._n_crop
      },
      n_sample = nSample
   }
end
