####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

# Creates the proper module file name from the project name.

# Inclusion settings
include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${FALCON_INCLUDE_DIRS}
   ${CURL_INCLUDE_DIR}
)
# Enable this include if the project has a private include
#include_directories( ${PROJECT_SOURCE_DIR}/include )


# Sources files the module is built from.
set(SRC_FILES 
   curl.cpp
   curl_ext.cpp
   curl_mod.cpp
   curl_fm.cpp
   curl_st.cpp
   # add more source files starting here:
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
   curl_ext.h
   curl_mod.h
   curl_st.h
   curl_fm.h
   version.h
   # add more header files starting here:
)

# Builds and links from the source files
set(SOURCES
   ${SRC_FILES}
   ${HDR_FILES} # optional but useful, see comment above.
)

FalconBuildCanonical( curl "${SOURCES}" "${CURL_LIBRARY}" "" ) 

