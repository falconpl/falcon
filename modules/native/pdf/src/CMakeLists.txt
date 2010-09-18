include_directories(
  ${Falcon_INCLUDE_DIRS}
  ${LibHaru_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(scriptExtensions_SRCS
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
   scriptExtensions/linkannotation.cpp
   scriptExtensions/error.cpp
)
add_library(hpdf_fm MODULE
   hpdf.cpp
   moduleImpl/st.cpp
   moduleImpl/doc.cpp
   moduleImpl/encoder.cpp
   moduleImpl/encoder_unicode.cpp
   moduleImpl/error.cpp
   moduleImpl/array.cpp
   moduleImpl/dict.cpp
   
   ${scriptExtensions_SRCS}
)

target_link_libraries(hpdf_fm
  ${LibHaru_LIBRARIES} 
)
falcon_finalize_module(hpdf_fm)

## prepare some variables for usage in other parts of this project
set(_tmp ${scriptExtensions_SRCS})
set(scriptExtensions_SRCS)
foreach(item ${_tmp})
  list(APPEND scriptExtensions_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${item})
endforeach(item)
set(scriptExtensions_SRCS ${scriptExtensions_SRCS} PARENT_SCOPE) # bubble it up