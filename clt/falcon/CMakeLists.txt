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
   if(NOT MINGW)
      set( SYSTEM_RC falcon.rc)
   endif(NOT MINGW)
else()  
   if(FALCON_READLINE_MODE STREQUAL "readline")
     set(readline_SRC readline_readline.cpp)
     add_definitions( -DFALCON_WITH_GPL_READLINE )

      FIND_PATH(READLINE_INCLUDE_DIR readline/readline.h)
      FIND_LIBRARY(READLINE_LIBRARY NAMES readline)

      IF (READLINE_INCLUDE_DIR AND READLINE_LIBRARY)
         SET(READLINE_FOUND TRUE)
         MESSAGE(STATUS "Found Readline library: ${READLINE_LIBRARY}")
         MESSAGE(STATUS "Include dir is: ${READLINE_INCLUDE_DIR}")
         INCLUDE_DIRECTORIES(${READLINE_INCLUDE_DIR})
      ELSE (READLINE_INCLUDE_DIR AND READLINE_LIBRARY)
         SET(READLINE_FOUND FALSE)
         MESSAGE(FATAL_ERROR "** Readline library not found!\n** Your distro may provide a binary for Readline e.g. for ubuntu try apt-get install libreadline5-dev")
      ENDIF (READLINE_INCLUDE_DIR AND READLINE_LIBRARY)

      include_directories( ${READLINE_INCLUDE_DIR} )
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
   int_mode.cpp
   options.cpp
   testmode.cpp
   ${readline_SRC}
   ${SYSTEM_RC}
)

target_link_libraries(falcon 
  falcon_engine
  ${Editline_LIBRARY}
  ${READLINE_LIBRARY}
)

install( TARGETS falcon DESTINATION ${FALCON_INSTALL_DESTINATIONS})
