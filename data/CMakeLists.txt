FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/data "${src}" "${luasrc}")
