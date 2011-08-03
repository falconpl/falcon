##################################################
# Falcon Programming Language
#  -- New engine tests.
##################################################

include_directories(
  include
  ../include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

message(STATUS "Compiling tests for new engine on ${CMAKE_SYSTEM_NAME}" )

###############################################
message("Adding debugtest")
add_executable( debugtest debugtest.cpp )
target_link_libraries( debugtest falcon_engine )

###############################################
message("Adding looptest")
add_executable( looptest looptest.cpp )
target_link_libraries( looptest falcon_engine )

###############################################
message("Adding fib")
add_executable( fib fib.cpp )
target_link_libraries( fib falcon_engine )

###############################################
message("Adding fact")
add_executable( fact fact.cpp )
target_link_libraries( fact falcon_engine )

###############################################
message("Adding stringtest")
add_executable( stringtest stringtest.cpp )
target_link_libraries( stringtest falcon_engine )

###############################################
message("Adding and_test")
add_executable( and_test and_test.cpp )
target_link_libraries( and_test falcon_engine )

###############################################
message("Adding ruletest")
add_executable( ruletest ruletest.cpp )
target_link_libraries( ruletest falcon_engine )

###############################################
message("Adding testparser")
add_executable( testparser testparser.cpp )
target_link_libraries( testparser falcon_engine )

###############################################
message("Adding interactive mode")
add_executable( intmode intmode.cpp )
target_link_libraries( intmode falcon_engine )

###############################################
message("Adding a test module")
add_library( testmod_fm MODULE
   testmod.cpp )

set_target_properties( 
   testmod_fm
   PROPERTIES 
      PREFIX "" )

target_link_libraries( testmod_fm falcon_engine )