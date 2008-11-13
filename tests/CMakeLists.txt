####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib - TESTS
####################################################################

ADD_LIBRARY( tests_dynlib MODULE
   testsrc/test_dynlib.c
)

IF(APPLE)
   SET_TARGET_PROPERTIES(tests_dynlib
      PROPERTIES
         PREFIX ""
         SUFFIX ".dylib" )
ELSE(APPLE)
   SET_TARGET_PROPERTIES(tests_dynlib
      PROPERTIES PREFIX "")
ENDIF(APPLE)

#installation of the dynlib goes in the testsuite source.
#this is because we need it for our tests.
INSTALL( TARGETS tests_dynlib
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/testsuite )
