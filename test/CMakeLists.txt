FILE(GLOB luasrc *.lua Lenna.png)
ADD_TORCH_PACKAGE(dp/test "${src}" "${luasrc}")
