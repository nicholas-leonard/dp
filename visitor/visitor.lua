------------------------------------------------------------------------
--[[ Visitor ]]--
-- Visits a composite struture of Models and modifies their states.
-- Visitors should try to access a model method assigned to 
-- each visitor (if itexists). This would allow models to implement
-- visitor specifics. (already started with dp.MaxNorm model)
------------------------------------------------------------------------
local Visitor = torch.class("dp.Visitor")
Visitor.isVisitor = true

function Visitor:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, name, zero_grads, include, exclude, observer, verbose 
      = xlua.unpack(
      {config},
      'Visitor', 
      'Visits a composite struture of Models and modifies their states.',
      {arg='name', type='string', req=true,
       help='identifies visitor in reports.'},
      {arg='zero_grads', type='boolean', default=true,
       help='calls zeroGradParameters on each model after it has '..
       'been visited. Should be true for the root vistior.'..
       'Note that the VisitorChain sets its components to false'},
      {arg='include', type='table',
       help='only models having a true value for the member named ' .. 
       'in this table are visited, unless the member is also listed ' ..
       'in the exclude table, in this case it is not visited. ' ..
       'If include is empty, all models are included, unless ' ..
       'specified in the exclude list'},
      {arg='exclude', type='table', default='',
       help='models having a member named in this table are not ' ..
       'visited, even if the member is in the include table, ' ..
       'i.e. exclude has precedence over include. Defaults to {}'},
      {arg='observer', type='dp.Observer', 
       help='observer that is notified when an event occurs.'},
      {arg='verbose', type='boolean', default=true,
       help='can print messages to stdout'}
   )
   self._name = name
   self._zero_grads = zero_grads
   self._exclude = (exclude == '') and {} or exclude
   self._include = include
   self._verbose = verbose
   self:setObserver(observer)
end

function Visitor:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, model, propagator, id = xlua.unpack(
      {config},
      'Visitor:setup', 
      'Visitor post-initialization method',
      {arg='mediator', type='dp.Mediator',
       help='used for inter-object communication.'},
      {arg='model', type='dp.Model',
       help='model that will be visited'},
      {arg='propagator', type='dp.Propagator',
       help='the propagator which makes use of this visitor'},
      {arg='id', type='dp.ObjectID',
       help='Set automatically by propagator. Use only for unit tests'}
   )
   self._mediator = mediator
   -- not sure including model is good idea...
   self._model = model
   self._propagator = propagator
   self._id = id or propagator:id():create(self._name)
   self._name = nil
   if self._observer then
      self._observer:setup{mediator=mediator, subject=self}
   end
end

function Visitor:id()
   return self._id
end

function Visitor:name()
   return self._id:name()
end

function Visitor:setObserver(observer)
   if not torch.typename(observer) and type(observer) == 'table' then
      --if list, make composite observer
      observer = dp.CompositeObserver(observer)
   end
   self._observer = observer
end

function Visitor:observer()
   return self._observer
end

-- compares model to filter to see if it can be visited
function Visitor:canVisit(model)
   local model_tags = model:tags()
   if self._exclude and not _.isEmpty(self._exclude) then
      for i, tag in ipairs(self._exclude) do
         if model_tags[tag] then
            return false
         end
      end
   end
   if self._include and not _.isEmpty(self._include) then
      for i, tag in ipairs(self._include) do
         if model_tags[tag] then
            return true
         end
      end
   else
      return true
   end
   return false
end

function Visitor:visitModel(model)
   -- can we visit model?
   if not self:canVisit(model) then 
      return 
   end
   
   if not model.mvstate[self:id():name()] then 
      model.mvstate[self:id():name()] = {}
   end
   
   self:_visitModel(model)
   
   self:doneVisit(model)
end

function Visitor:doneVisit(model)
   if self._zero_grads then
      model:zeroGradParameters()
   end
end

function Visitor:_visitModel(model)
   return
end

--default is to do nothing for visitors (for now)
function Visitor:visitContainer(model)
   
end

function Visitor:report()
   return {[self:name()] = {}}
end

function Visitor:setZeroGrads(zero_grads)
   self._zero_grads = zero_grads
end

