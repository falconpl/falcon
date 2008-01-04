####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

PROJECT(PDF)
SET(FALCON_PROJECT_BASE "modules/pdf")

#Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

#Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")
#set common include directory
FALCON_INCLUDE_SETTINGS()

INCLUDE_DIRECTORIES(BEFORE include)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

# Target
ADD_LIBRARY( pdf MODULE
   # PDF module source
   pdf.cpp
   pdf_ext.cpp
   pdf_srv.cpp
)

#Link
TARGET_LINK_LIBRARIES(pdf falcon_engine hpdf z)
SET_TARGET_PROPERTIES(pdf PROPERTIES PREFIX "") #Install
FALCON_MODULE_INSTALL( pdf )
