####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE curl )

# Inclusion settings
include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Falcon_INCLUDE_DIRS}
   ${CURL_INCLUDE_DIR}
)
# Enable this include if the project has a private include
#include_directories( ${PROJECT_SOURCE_DIR}/include )


# Sources files the module is built from.
set(SRC_FILES 
   curl.cpp
   curl_ext.cpp
   curl_mod.cpp
   curl_st.cpp
   # add more source files starting here:
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
   curl_ext.h
   curl_mod.h
   curl_st.h
   version.h
   # add more header files starting here:
)

# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional but useful, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} 
   falcon_engine
   ${CURL_LIBRARY}
)

falcon_install_module( ${CURRENT_MODULE} )
