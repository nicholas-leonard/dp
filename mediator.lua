------------------------------------------------------------------------
--[[ Subscriber ]]--
-- Used by Mediator. Holds a subscriber object which will be called
-- back via its func_name method name. Since no functions are being 
-- pointed to directly, the object can be serialized...
------------------------------------------------------------------------
local Subscriber = torch.class("dp.Subscriber")
Subscriber.isSubscriber = true

function Subscriber:__init(subscriber, func_name, id, options)
   self.options = options or {}
   self.subscriber = subscriber
   self.func_name = func_name
   self.channel = nil
   self.id = id --or math.random(1000000000)
end

function Subscriber:update(options)
   if options then
      self.subscriber = options.subscriber or self.subscriber
      self.func_name = options.func_name or self.func_name
      self.options = options.options or self.options
   end
end

------------------------------------------------------------------------
--[[ Channel ]]--
-- Used by Mediator. Can be published and subscribed to.
------------------------------------------------------------------------
local Channel = torch.class("dp.Channel")
Channel.isChannel = true

function Channel:__init(namespace, parent)
    self.stopped = false
    self.namespace = namespace
    self.callbacks = {}
    self.channels = {}
    self.parent = parent
end

function Channel:addSubscriber(subscriber, func_name, id, options)
   local callback = dp.Subscriber(subscriber, func_name, id, options)
   local priority = (#self.callbacks + 1)

   options = options or {}

   if options.priority and
     options.priority >= 0 and
     options.priority < priority
   then
       priority = options.priority
   end

   table.insert(self.callbacks, priority, callback)

   return callback
end

function Channel:getSubscriber(id)
   for i=1, #self.callbacks do
     local callback = self.callbacks[i]
     if callback.id == id then return { index = i, value = callback } end
   end
   local sub
   for _, channel in pairs(self.channels) do
     sub = channel:getSubscriber(id)
     if sub then break end
   end
   return sub
end

function Channel:setPriority(id, priority)
   local callback = self:getSubscriber(id)

   if callback.value then
     table.remove(self.callbacks, callback.index)
     table.insert(self.callbacks, priority, callback.value)
   end
end

function Channel:addChannel(namespace)
   self.channels[namespace] = dp.Channel(namespace, self)
   return self.channels[namespace]
end

function Channel:hasChannel(namespace)
   return namespace and self.channels[namespace] and true
end

function Channel:getChannel(namespace)
   return self.channels[namespace] or self:addChannel(namespace)
end

function Channel:removeSubscriber(id)
   local callback = self:getSubscriber(id)

   if callback and callback.value then
     for _, channel in pairs(self.channels) do
       channel:removeSubscriber(id)
     end

     return table.remove(self.callbacks, callback.index)
   end
end

function Channel:publish(channelNamespace, ...)
   for i = 1, #self.callbacks do
      local callback = self.callbacks[i]
      -- if it doesn't have a predicate, or it does and it's true then run it
      if not callback.options.predicate or callback.options.predicate(...) then
         --print(torch.typename(callback.subscriber), callback.func_name)
         callback.subscriber[callback.func_name](callback.subscriber, ...)
      end
   end
end

------------------------------------------------------------------------
--[[ Mediator ]]--
-- An object oriented mediator. Callbacks are methods specified by 
-- an object and a method name. 
------------------------------------------------------------------------
local Mediator = torch.class("dp.Mediator")
Mediator.isMediator = true

function Mediator:__init()
   self.channel = dp.Channel('root')
   self.id_gen = 0
end

function Mediator:nextId()
   local id_gen = self.id_gen 
   self.id_gen = self.id_gen + 1
   return id_gen
end

function Mediator:getChannel(channelNamespace)
   if channelNamespace == ':' then
      return self.channel
   end
   
   if type(channelNamespace) == 'string' then
      channelNamespace = _.split(channelNamespace, ':')
   end
   
   local channel = self.channel

   for i=1, #channelNamespace do
      channel = channel:getChannel(channelNamespace[i])
   end

   return channel
end

function Mediator:subscribe(channelNamespace, subscriber, func_name, options)
   local id = self:nextId()
   local channel = self:getChannel(channelNamespace)
   return channel:addSubscriber(subscriber, func_name, id, options)
end

function Mediator:getSubscriber(id, channelNamespace)
   return self:getChannel(channelNamespace):getSubscriber(id)
end

function Mediator:removeSubscriber(id, channelNamespace)
   return self:getChannel(channelNamespace):removeSubscriber(id)
end

function Mediator:publish(channelNamespace, ...)
   local channel = self:getChannel(channelNamespace)
   channel:publish(channelNamespace, ...)
end
