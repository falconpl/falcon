include_directories(
  ${Falcon_INCLUDE_DIRS}
  ${LibHaru_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(hpdf_fm MODULE
   hpdf.cpp
   moduleImpl/st.cpp
   moduleImpl/doc.cpp
   moduleImpl/encoder.cpp
   moduleImpl/error.cpp
   moduleImpl/array.cpp
   moduleImpl/dict.cpp
   scriptExtensions/consts.cpp
   scriptExtensions/enums.cpp
   scriptExtensions/doc.cpp
   scriptExtensions/page.cpp
   scriptExtensions/font.cpp
   scriptExtensions/destination.cpp
   scriptExtensions/image.cpp
   scriptExtensions/outline.cpp
   scriptExtensions/encoder.cpp
   scriptExtensions/textannotation.cpp
   #scriptExtensions/xobject.cpp
   scriptExtensions/error.cpp
)

target_link_libraries(hpdf_fm
  ${LibHaru_LIBRARIES} 
)
falcon_finalize_module(hpdf_fm)
