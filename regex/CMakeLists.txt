####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Regex
####################################################################

IF( WIN32 OR APPLE )
   OPTION(WITH_INTERNAL_PCRE "Use internal pcre library" ON)
ELSE( WIN32 OR APPLE )
   OPTION(WITH_INTERNAL_PCRE "Use internal pcre library" OFF)
ENDIF( WIN32 OR APPLE)

# Inclusion settings
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES( ${TARGET_DIR}/${TARGET_LIB_DIR} )

IF(WIN32)
   SET( SYS_RC regex.rc)
ENDIF(WIN32)

IF(WITH_INTERNAL_PCRE)
   #include local headers BEFORE
   INCLUDE_DIRECTORIES( BEFORE "." )

   #Compile our version of PCRE...
   SET(PCRE_SOURCE
      pcre_chartables.c
      pcre_compile.c
      pcre_config.c
      pcre_exec.c
      pcre_fullinfo.c
      pcre_get.c
      pcre_globals.c
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

   
   ADD_DEFINITIONS( -DHAVE_CONFIG_H="1" -DPCRE_EXP_DECL="" -DPCRE_EXP_DATA_DECL="extern" -DPCRE_EXP_DEFN=" " -DPCRE_EXP_DATA_DEFN=" ")
   SET_SOURCE_FILES_PROPERTIES(
      ${PCRE_SOURCE}
      PROPERTIES LANGUAGE "CXX"
   )

   SET(PCRE_LIBRARIES "")

ELSE(WITH_INTERNAL_PCRE)
   #We'll activate this when CMAKE will offer a standard FINDPCRE macro
   #IF(NOT PCRE_FOUND AND NOT PCRE_LIBRARIES)
      IF(NOT PCRE_LIB_NAMES)
         SET(PCRE_LIB_NAMES pcre)
      ENDIF(NOT PCRE_LIB_NAMES)

      IF(NOT PCRE_INCLUDE_SEARCH_PATH)
         SET(PCRE_INCLUDE_SEARCH_PATH /usr/include /usr/local/include )
      ENDIF(NOT PCRE_INCLUDE_SEARCH_PATH)

      IF(NOT PCRE_LIB_SEARCH_PATH)
          SET(PCRE_LIB_SEARCH_PATH /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64)
      ENDIF(NOT PCRE_LIB_SEARCH_PATH)


      FIND_PATH(PCRE_INCLUDE_DIR pcre.h ${PCRE_INCLUDE_SEARCH_PATH})
      FIND_LIBRARY(PCRE_LIBRARIES
         NAMES ${PCRE_LIB_NAMES}
         PATHS ${PCRE_LIB_SEARCH_PATH}
      )
   #ENDIF(NOT PCRE_FOUND AND NOT PCRE_LIBRARIES)

   IF(PCRE_INCLUDE_DIR AND PCRE_LIBRARIES)
      INCLUDE_DIRECTORIES(BEFORE ${PCRE_REAL_INCLUDE_DIR})
   ELSE(PCRE_INCLUDE_DIR AND PCRE_LIBRARIES)
      IF(NOT PCRE_INCLUDE_DIR)
         MESSAGE(SEND_ERROR "PCRE include path was not found")
         MESSAGE(SEND_ERROR "Looked in: ${PCRE_INCLUDE_SEARCH_PATH}")
      ENDIF(NOT PCRE_INCLUDE_DIR)
      IF(NOT PCRE_LIBRARIES)
         MESSAGE(SEND_ERROR "PCRE library was not found")
         MESSAGE(SEND_ERROR "Looked for: ${PCRE_LIB_NAMES}")
         MESSAGE(SEND_ERROR "Looked in: ${PCRE_LIB_SEARCH_PATH}")
      ENDIF(NOT PCRE_LIBRARIES)
      MESSAGE(SEND_ERROR "Could not find PCRE, you can:")
      MESSAGE(SEND_ERROR "Specify: -DWITH_INTERNAL_PCRE=ON")
      MESSAGE(SEND_ERROR "Or use:")
      MESSAGE(SEND_ERROR "-DPCRE_INCLUE_DIR=PATH_TO_PCRE_INCLUDES")
      MESSAGE(FATAL_ERROR "-DPCRE_LIBRARIES=PATH_TO_PCRE_LIBRARY")
   ENDIF(PCRE_INCLUDE_DIR AND PCRE_LIBRARIES)
   
   # Force ignoring OUR pcre
   INCLUDE_DIRECTORIES(BEFORE ${PCRE_INCLUDE_DIR} ) 
ENDIF(WITH_INTERNAL_PCRE)

# Target
ADD_LIBRARY( regex_fm MODULE
${PCRE_SOURCE}

regex.cpp
regex_ext.cpp
regex_mod.cpp
regex_st.cpp

${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(regex_fm falcon_engine ${PCRE_LIBRARIES})

#Install
FALCON_INSTALL_MODULE( regex_fm )
