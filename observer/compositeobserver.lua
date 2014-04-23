------------------------------------------------------------------------
--[[ CompositeObserver ]]--
-- Is composed of multiple observers
------------------------------------------------------------------------
local CompositeObserver = torch.class("dp.CompositeObserver", "dp.Observer")
CompositeObserver.isCompositeObserver = true

function CompositeObserver:__init(observers)
   self._observers = observers
   for name, observer in pairs(self._observers) do
      assert(observer.isObserver)
   end
   for _, observer in ipairs(self._observers) do
      assert(observer.isObserver)
   end
end

function CompositeObserver:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, subject = xlua.unpack(
      {config},
      'Observer:setup', nil,
      {arg='mediator', type='dp.Mediator', req=true},
      {arg='subject', type='dp.Experiment | dp.Propagator | ...',
      help='object being observed.'}
   )
   assert(mediator.isMediator)
   self._mediator = mediator
   self:setSubject(subject)
   for name, observer in pairs(self._observers) do
      observer:setup(config)
   end
end
