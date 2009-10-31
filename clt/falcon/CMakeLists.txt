##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################

if(WIN32)
   set(readline_SRC readline_simple.cpp)
   set( SYSTEM_RC falcon.rc)
else()  
   option(FALCON_WITH_INTERNAL_EDITLINE "" OFF)
   if(FALCON_WITH_INTERNAL_EDITLINE)
     add_subdirectory(editline)
     set(Editline_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/editline/src)
     set(Editline_LIBRARY edit)
   else()
     find_package(Editline REQUIRED)
   endif()
   set(readline_SRC readline_editline.cpp)
endif(WIN32)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${Editline_INCLUDE_DIRS}
  
)

add_executable( falcon
   falcon.cpp
   int_mode.cpp
   options.cpp
   ${readline_SRC}
   ${SYSTEM_RC}
)

target_link_libraries(falcon 
  falcon_engine 
  ${Editline_LIBRARY}
)

install( TARGETS falcon ${FALCON_INSTALL_DESTINATIONS})
