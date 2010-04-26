include_directories(
  ${Falcon_INCLUDE_DIRS}
  ${LibHaru_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(hpdf_fm MODULE
   hpdf.cpp
   moduleImpl/st.cpp
   moduleImpl/doc.cpp
   moduleImpl/page.cpp
   moduleImpl/font.cpp
   moduleImpl/error.cpp
   scriptExtensions/doc.cpp
   scriptExtensions/page.cpp
   scriptExtensions/font.cpp
   scriptExtensions/error.cpp
)

target_link_libraries(hpdf_fm
  ${LibHaru_LIBRARIES} 
)
falcon_finalize_module(hpdf_fm)
