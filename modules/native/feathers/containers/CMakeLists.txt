####################################################################
# The Falcon Programming Language
#
# CMake configuration file for containers (List, Set &c)
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE containers )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32 AND NOT MINGW)
   set( SYS_RC containers.rc)
endif()

# Sources files the module is built from.
set(SRC_FILES 
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

# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)

falcon_install_module( ${CURRENT_MODULE} )

