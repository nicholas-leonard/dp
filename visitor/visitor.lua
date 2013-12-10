------------------------------------------------------------------------
--[[ Visitor ]]--
-- Visits a composite struture of Models and modifies their states.

-- TODO: 
-- Visitors should try to access a model method assigned to 
-- each visitor (if exists). This would allow models to implement
-- visitor specifics. (already started with dp.Linear model)
-- Visitors accumulate statistics for reporting purposes
-- Visitor statistics
------------------------------------------------------------------------
local Visitor = torch.class("dp.Visitor")

function Visitor:__init(...)
   local args, name, include, exclude, observer = xlua.unpack(
      {... or {}},
      'Visitor', nil,
      {arg='name', type='string', req=true,
       help='identifies visitor in reports.'},
      {arg='include', type='table',
       help='only models having a true value for the member named ' .. 
       'in this table are visited, unless the member is also listed ' ..
       'in the exclude table, in this case it is not visited. ' ..
       'If include is empty, all models are included, unless ' ..
       'specified in the exclude list'},
      {arg='exclude', type='table', default={},
       help='models having a member named in this table are not ' ..
       'visited, even if the member is in the include table, i.e. ' ..
       'exclude has priority over include'},
      {arg='observer', type='dp.Observer', 
       help='observer that is informed when an event occurs.'}
   )
   self._name = name
   self._exclude = exclude
   self._include = include
   self:setObserver(observer)
end

function Visitor:setup(...)
   local args, mediator, model, propagator = xlua.unpack(
      {... or {}},
      'Visitor:setup', nil,
      {arg='mediator', type='dp.Mediator'},
      {arg='model', type='dp.Model'},
      {arg='propagator', type='dp.Propagator'}
   )
   self._mediator = mediator
   -- not sure including model is good idea...
   self._model = model
   self._propagator = propagator
   self._id = propagator:id():create(self._name)
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
      for tag in ipairs(self._exclude) do
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
   --TODO : mvstate[self:id():parent():name()][self:name()]
   -- or mvstate[self._id_string] where self._id_string = self._id:toString())
   -- has the model-visitor state been initialized?
   if not model.mvstate[self:id():name()] then 
      model.mvstate[self:id():name()] = {}
   end
   self:_visitModel(model)
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

