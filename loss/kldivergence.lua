------------------------------------------------------------------------
--[[ KLDivergence ]]--
-- Loss subclass
-- Adapter of nn.nn.DistKLDivCriterion
-- KL-divergence 
------------------------------------------------------------------------
local KLDivergence, parent = torch.class("dp.KLDivergence", "dp.Loss")
KLDivergence.isKLDivergence = true

function KLDivergence:__init(config)
   self._criterion = nn.DistKLDivCriterion()
   config = config or {}
   config.target_view = config.target_view or 'bf'
   config.input_view = config.input_view or 'bf'
   parent.__init(self, config)
end
