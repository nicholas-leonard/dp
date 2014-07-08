FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/propagator "${src}" "${luasrc}")
