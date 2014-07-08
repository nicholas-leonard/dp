FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/observer "${src}" "${luasrc}")
