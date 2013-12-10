------------------------------------------------------------------------
--[[ CompositeObserver ]]--
-- Is composed of multiple observers
------------------------------------------------------------------------

local CompositeObserver = torch.class("dp.CompositeObserver", "dp.Observer")

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
   local args, mediator, subject = xlua.unpack(
      {config or {}},
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

function CompositeObserver:report()
   error"NotSupported : observers don't generate reports"
   --[[local report = {}
   for name, observer in pairs(self._observers) do
      assert(observers)
      local observer_report = observer:report()
      if observer_report and not _.isEmpty(observer_report)  then
         report[name] = observer_report
      end
   end
   return report]]--
end
