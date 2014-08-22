a = 7

c1 = coroutine.create(function()
   a = 1
   coroutine.yield()
   print(1, a)
end)

c2 = coroutine.create(function()
   a = 2
   coroutine.yield()
   print(2, a)
end)

coroutine.resume(c1)
coroutine.resume(c2)
coroutine.resume(c1)
coroutine.resume(c2)
