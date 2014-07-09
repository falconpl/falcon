####################################################################
# The Falcon Programming Language
#
# CMake configuration file for vm (Virtual Machine interface)
####################################################################

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set( SYS_RC vm.rc)
endif()

# Sources files the module is built from.
set(SRC_FILES 
  vm.cpp
  vm_fm.cpp
  vm_ext.cpp
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  version.h
  vm_fm.h
  vm_ext.h
)

# Builds and links from the source files
set( SOURCES
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

FalconDeclareFeather( "vm" "${SOURCES}" "" )
