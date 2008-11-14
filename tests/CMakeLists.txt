####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib - TESTS
####################################################################

INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

ADD_LIBRARY( test_dynlib MODULE
   testsrc/test_dynlib.c
)

IF(APPLE)
   SET_TARGET_PROPERTIES(test_dynlib
      PROPERTIES
         PREFIX ""
         SUFFIX ".dylib" )
ELSE(APPLE)
   SET_TARGET_PROPERTIES(test_dynlib
      PROPERTIES PREFIX "")
ENDIF(APPLE)

#installation of the dynlib goes in the testsuite source.
#this is because we need it for our tests.
INSTALL( TARGETS test_dynlib
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/testsuite )
