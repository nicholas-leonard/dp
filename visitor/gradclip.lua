------------------------------------------------------------------------
--[[ GradClip ]]--
-- Ref.: A. http://goo.gl/Zxza8m
-- B. http://jmlr.org/proceedings/papers/v28/pascanu13.pdf
-- Visitor
-- Hard constraint on the upper bound of the norm of gradient with 
-- respect to parameters (gradParams). Unlike ref A and B, which apply
-- the constraint on the norm of all parameters, the norm is applied 
-- on the norm of each Layer's parameters.
-- Should occur before Learn in VisitorChain
------------------------------------------------------------------------
local GradClip, parent = torch.class("dp.GradClip", "dp.Visitor")
GradClip.isGradClip = true

function GradClip:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, cutoff, name = xlua.unpack(
      {config},
      'GradClip', 
      'Hard constraint on the upper bound of the norm of gradParams.',
      {arg='cutoff', type='number', default=1,
       help="max norm of a Layer's parameters"},
      {arg='name', type='string', default='gradclip',
       help='identifies visitor in reports.'}
   )
   self._cutoff = cutoff
   config.include = config.include or {}
   table.insert(config.include, 'hasParams')
   config.exclude = config.exclude or {}
   table.insert(config.exclude, 'no-gradclip')
   config.name = name
   parent.__init(self, config)
   self.norms = {}
end

function GradClip:_visitModel(model)
   if model.gradClip then
      local norm = model:gradClip(self._cutoff)
      -- keep a moving average of norms 
      self.norms[model:id():toString()] = (self.norms[model:id():toString()] or 0)*0.8 + norm*0.2
   else
      if not model.mvstate[self:id():name()].warned then
         print("Warning: GradClip not implemented for model " .. 
            torch.typename(model) .. ". Ignoring model-visitor pair")
         model.mvstate[self:id():name()].warned = true
      end
   end
end

function GradClip:report()
   local norms = _.values(self.norms)
   if self._verbose then
      print(self:id():name().." norms: ", unpack(norms))
   end
   local report = {
      [self:name()] = {
         norms = self.norms
      }
   }
   return report
end
