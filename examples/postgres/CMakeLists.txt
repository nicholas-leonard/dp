FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/examples/postgres "${src}" "${luasrc}")
