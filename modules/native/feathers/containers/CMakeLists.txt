####################################################################
# The Falcon Programming Language
#
# CMake configuration file for containers (List, Set &c)
####################################################################

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set( SYS_RC containers.rc)
endif()

# Sources files the module is built from.
set(SRC_FILES 
  containers.cpp
  containers_fm.cpp
  classcontainer.cpp
  classiterator.cpp
  classlist.cpp
  container_mod.cpp
  iterator_mod.cpp
  list_mod.cpp
  
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  version.h
  containers_fm.h
  classcontainer.h
  classiterator.h
  classlist.h
  container_mod.h
  iterator_mod.h
  list_mod.h
  errors.h
)

set( SOURCES
   ${SRC_FILES}
   ${HDR_FILES} 
)

FalconDeclareFeather( "containers" "${SOURCES}" "" )
