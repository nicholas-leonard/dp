------------------------------------------------------------------------
--[[ BoundingBoxDetect ]]--
-- You have two overlapping bounding boxes. You compute the 
-- intersection of the boxes, which is the area of the overlap. 
-- You compute the uniton of the overlapping boxes, which is the 
-- sum of the areas of the entrie boxes minus the area of the overlap. 
-- Then you divide the intersection by the union.
------------------------------------------------------------------------
local BBD, parent = torch.class("dp.BoundingBoxDetect", "dp.Feedback")

function BBD:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, bce, name, output_module = xlua.unpack(
      {config},
      'Confusion', 
      'Adapter for optim.ConfusionMatrix',
      {arg='bce', type='boolean', default=false,
       help='set true when using Binary Cross-Entropy (BCE)Criterion'},
      {arg='name', type='string', default='confusion',
       help='name identifying Feedback in reports'},
      {arg='output_module', type='nn.Module',
       help='module applied to output before measuring confusion matrix'}
   )
   config.name = name
   self._bce = bce
   self._output_module = output_module or nn.Identity()
   parent.__init(self, config)
end
