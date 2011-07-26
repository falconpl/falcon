##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################


SET (FALCON_READLINE_MODE "None" CACHE STRING 
         "Use 'readline' (GNU systems only), 'editline' or 'none' when interactive.")
option(FALCON_WITH_INTERNAL_EDITLINE "If using Editline, compile the one internally provided." ON)

if(WIN32)
   set(readline_SRC readline_simple.cpp)
   set( SYSTEM_RC falcon.rc)
else()  
   if(FALCON_READLINE_MODE STREQUAL "readline")
     set(readline_SRC readline_editline.cpp)
     add_definitions( -DFALCON_WITH_GPL_READLINE )
   else()
      if( FALCON_READLINE_MODE STREQUAL "editline" )
         set(readline_SRC readline_editline.cpp)
         if(FALCON_WITH_INTERNAL_EDITLINE)
            add_subdirectory(editline)
            set(Editline_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/editline/src)
            set(Editline_LIBRARY edit)
         else()
            find_package(Editline REQUIRED)
         endif()
      else()
         set(readline_SRC "readline_simple.cpp")
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
   #int_mode.cpp
   options.cpp
   ${readline_SRC}
   ${SYSTEM_RC}
)

target_link_libraries(falcon 
  falcon_engine
  ${Editline_LIBRARY}
)

install( TARGETS falcon ${FALCON_INSTALL_DESTINATIONS})
