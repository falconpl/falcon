include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${falcon_INCLUDE_DIRS}
  ${LibHaru_INCLUDE_DIRS}
)


# Target
add_library(hpdf_fm MODULE
   hpdf.cpp
   hpdf_ext.cpp
   hpdf_mod.cpp
)

target_link_libraries(hpdf_fm
  ${LibHaru_LIBRARIES} 
)
falcon_finalize_module(hpdf_fm)