------------------------------------------------------------------------
--[[ SequenceView ]]-- 
-- A DataView holding a tensor of sequences.
-- Has view = bwc 
-- Output of nn.LookupTable, input of nn.Temporal*
------------------------------------------------------------------------
local SequenceView, parent = torch.class("dp.SequenceView", "dp.DataView")
SequenceView.isSequenceView = true

-- batch x width x channels
-- used for temporal convolutions
function SequenceView:bwc()
   if #self._view ~= 3 then 
      error("Cannot convert view '"..self._view.."' to 'bwc'")
   end
   if self._view == 'bwc' then
      return nn.Identity()
   end
   return self:transpose('bwc')
end

-- batch x channels x width
function SequenceView:bcw()
   if #self._view ~= 3 then 
      error("Cannot convert view '"..self._view.."' to 'bcw'")
   end
   if self._view == 'bcw' then
      return nn.Identity()
   end
   return self:transpose('bcw')
end
