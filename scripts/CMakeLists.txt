FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/scripts "${src}" "${luasrc}")
