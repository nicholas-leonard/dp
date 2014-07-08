FILE(GLOB luasrc *.lua)
ADD_TORCH_PACKAGE(dp/preprocess "${src}" "${luasrc}")
