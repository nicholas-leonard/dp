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
   
end
