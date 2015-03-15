FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/sampler "${src}" "${luasrc}")
