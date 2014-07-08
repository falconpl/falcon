##
#   Falcon MongoDB CMake file
#
#   Author(s): Stanislas Marquis <stnsls@gmail.com>
#

falcon_define_module( FALCON_MODULE mongo )

if(NOT FALCON_BUILD_MONGO )
	message( STATUS "NOT building module: mongo")
	return()
endif()
message( STATUS "Building module: mongo" )

##
#   Check for stdint.h
#
if ( NOT MSVC )
    find_file( STDINT_H stdint.h HINTS /usr/include )
    if ( STDINT_H STREQUAL STDINT_H-NOTFOUND )
        option( MONGO_HAVE_STDINT "MongoDB should use stdint.h" OFF )
    else()
        option( MONGO_HAVE_STDINT "MongoDB should use stdint.h" ON )
    endif()
    mark_as_advanced( STDINT_H )
    mark_as_advanced( MONGO_HAVE_STDINT )
endif()

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
#   Third-party pure C driver (mongoc)
#
add_subdirectory( src )

if ( MSVC )
    add_definitions( -DMONGO_USE__INT64 )
else()
    if ( MONGO_HAVE_STDINT )
        add_definitions( -DMONGO_HAVE_STDINT )
    else()
        add_definitions( -DMONGO_USE_LONG_LONG_INT )
    endif()
endif()

##
#   Target
#
add_library( ${FALCON_MODULE} MODULE
    mongodb_fm.cpp
    mongodb_ext.cpp
    mongodb_mod.cpp
    mongodb_srv.cpp
    mongodb_st.cpp )

add_dependencies( ${FALCON_MODULE} mongoc )

##
#   Linking
#
if ( WIN32 )
    target_link_libraries( ${FALCON_MODULE} falcon_engine mongoc wsock32 )
else()
    target_link_libraries( ${FALCON_MODULE} falcon_engine mongoc )
endif()

##
#   Install
#
falcon_install_module( ${FALCON_MODULE} )
