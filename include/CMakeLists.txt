####################################################################
# The Falcon Programming language
#
# Install header files for module development
###################################################################

configure_file( 
  falcon/config.h.in  ${CMAKE_CURRENT_BINARY_DIR}/falcon/config.h
  @ONLY 
)

file(GLOB falcon_files "falcon/*.h" )
list(APPEND falcon_files ${CMAKE_CURRENT_BINARY_DIR}/falcon/config.h)

install(FILES ${falcon_files} DESTINATION ${FALCON_INC_DIR}/falcon)
