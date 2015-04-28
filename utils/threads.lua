
function dp.Threads(...)
  local args = {...}
  -- https://github.com/torch/threads-ffi
  local Threads = require "threads"

  -- tell you if the queues are empty.
  function Threads:isEmpty()
    return not (self.mainqueue.head ~= self.mainqueue.tail or self.threadqueue.head ~= self.threadqueue.tail or self.endcallbacks.n > 0)
  end

  return Threads(unpack(args))
end
