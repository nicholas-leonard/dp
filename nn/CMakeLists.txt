FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/nn "${src}" "${luasrc}")
