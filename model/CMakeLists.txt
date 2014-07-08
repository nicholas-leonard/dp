FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/model "${src}" "${luasrc}")
