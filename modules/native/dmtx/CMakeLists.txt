##
#   Falcon DataMatrix CMake file
#
#   Author(s): Stanislas Marquis <stnsls@gmail.com>
#

if(NOT FALCON_BUILD_DMTX )
	message( STATUS "NOT building module: dmtx")
	return()
endif()
message( STATUS "Building module: dmtx" )

falcon_define_module( FALCON_MODULE dmtx )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

##
#   Find Falcon
#
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )
find_package( Falcon
    HINTS "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
          "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
          "$ENV{FALCON_BIN_PATH}/cmake"
    REQUIRED )

include_directories( ${Falcon_INCLUDE_DIRS} )

##
#   DataMatrix was previously found.
#
include_directories( BEFORE ${DMTX_INCLUDE_PATH} )

##
#   Target
#
add_library( ${FALCON_MODULE} MODULE
    dmtx_fm.cpp
    dmtx_ext.cpp
    dmtx_mod.cpp
    dmtx_srv.cpp
    #dmtx_st.cpp
    )

##
#   Linking
#
target_link_libraries( ${FALCON_MODULE} falcon_engine ${DMTX_LIBRARY} )

##
#   Install
#
falcon_install_module( ${FALCON_MODULE} )
