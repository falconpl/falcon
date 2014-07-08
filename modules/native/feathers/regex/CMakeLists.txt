####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Regex
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE regex )

# Find our Find*.cmake files 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)


if( WIN32 OR APPLE )
 set(_pcre_default ON)
else( WIN32 OR APPLE )
 set(_pcre_default OFF)
endif( WIN32 OR APPLE)
option(FALCON_WITH_INTERNAL_PCRE "Use internal pcre library" ${_pcre_default})

if(WIN32 AND NOT MINGW)
 set( SYS_RC regex.rc)
endif(WIN32 AND NOT MINGW)

if(FALCON_WITH_INTERNAL_PCRE)
 #include local headers BEFORE
 include_directories(${CMAKE_CURRENT_SOURCE_DIR})

 #Compile our version of PCRE...
 SET(PCRE_SOURCE
 pcre_chartables.c
 pcre_compile.c
 pcre_config.c
 pcre_exec.c
 pcre_fullinfo.c
 pcre_get.c
 pcre_info.c
 pcre_maketables.c
 pcre_newline.c
 pcre_ord2utf8.c
 pcre_refcount.c
 pcre_study.c
 pcre_tables.c
 pcre_try_flipped.c
 pcre_ucd.c
 pcre_valid_utf8.c
 pcre_version.c
 pcre_xclass.c
 )

 add_definitions(-DHAVE_CONFIG_H -DPCRE_STATIC -DPCRE_EXP_DATA_DECL=extern -DPCRE_EXP_DECL=extern -DPCRE_EXP_DEFN=)

 # This is an hack for systems where pcre is already hard-linked inside embedding c programs
 # Mangle names as C++, so it doesn't get the already-linked system pcre build.
 #if(NOT WIN32)
 set_source_files_properties(
 ${PCRE_SOURCE} PROPERTIES
 LANGUAGE "CXX"
 )
 #endif()

else()
 find_package(PCRE REQUIRED)
 # Force ignoring OUR pcre
 include_directories(${PCRE_INCLUDE_DIR})
endif()

add_library(${CURRENT_MODULE} MODULE
 regex.cpp
 regex_ext.cpp
 regex_mod.cpp
 ${PCRE_SOURCE}
 ${SYS_RC}
)

target_link_libraries(${CURRENT_MODULE}
 falcon_engine
 ${PCRE_LIBRARIES}
)

falcon_install_module( ${CURRENT_MODULE} )

set(_project_targets ${_project_targets} ${CURRENT_MODULE} PARENT_SCOPE)
