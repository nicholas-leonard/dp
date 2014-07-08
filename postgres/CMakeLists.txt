FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/postgres "${src}" "${luasrc}")
