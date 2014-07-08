FILE(GLOB luasrc *.lua)
ADD_SUBDIRECTORY(postgres)
ADD_TORCH_PACKAGE(dp/examples "${src}" "${luasrc}")
