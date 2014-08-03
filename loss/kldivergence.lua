------------------------------------------------------------------------
--[[ KLDivergence ]]--
-- Loss subclass
-- Adapter of nn.DistKLDivCriterion
-- KL-divergence 
------------------------------------------------------------------------
local KLDivergence, parent = torch.class("dp.KLDivergence", "dp.Loss")
KLDivergence.isKLDivergence = true

function KLDivergence:__init(config)
   self._criterion = nn.DistKLDivCriterion()
   config = config or {}
   -- criterion acts element-wise, so default view:
   config.target_view = config.target_view or 'default'
   config.input_view = config.input_view or 'default'
   parent.__init(self, config)
end
