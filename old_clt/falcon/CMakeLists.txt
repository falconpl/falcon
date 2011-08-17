##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################
option(FALCON_WITH_EDITLINE "Switch to ON for editline support." OFF)
option(FALCON_WITH_INTERNAL_EDITLINE "Switch to ON to use EDITLINE provided by falcon" ON)

if(WIN32)
   set(readline_SRC readline_simple.cpp)
   set( SYSTEM_RC falcon.rc)
else()  
   if(NOT FALCON_WITH_EDITLINE)
     set(readline_SRC readline_simple.cpp)
   else()
     set(readline_SRC readline_editline.cpp)
     if(FALCON_WITH_INTERNAL_EDITLINE)
       add_subdirectory(editline)
       set(Editline_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/editline/src)
       set(Editline_LIBRARY edit)
     else()
       find_package(Editline REQUIRED)
     endif()
  endif()
endif()

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
if(EDITLINE_FOUND AND FALCON_WITH_GPL_READLINE)
  set_target_properties(falcon PROPERTIES 
    COMPILE_DEFINITIONS FALCON_WITH_GPL_READLINE
  )
endif()
target_link_libraries(falcon 
  falcon_engine
  ${Editline_LIBRARY}
)

install( TARGETS falcon ${FALCON_INSTALL_DESTINATIONS})
