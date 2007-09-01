####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Regex
####################################################################
PROJECT(regex)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

IF(WIN32)
   SET( SYS_RC regex.rc)
ENDIF(WIN32)

# Target
ADD_LIBRARY( regex MODULE
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
pcre_ucp_searchfuncs.c
pcre_valid_utf8.c
pcre_version.c
pcre_xclass.c

regex.cpp
regex_ext.cpp
regex_mod.cpp
${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(regex falcon_engine)
SET_TARGET_PROPERTIES(regex
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( regex )
