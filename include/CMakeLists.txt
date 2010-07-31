####################################################################
# The Falcon Programming language
#
# Install header files for module development
###################################################################

configure_file( 
  falcon/config.h.in  ${CMAKE_CURRENT_BINARY_DIR}/falcon/config.h
  @ONLY 
)

file( GLOB falcon_files falcon/*.h)
file( GLOB falcon_srv falcon/srv/*.h)

set(FALCON_HEADERS
   ${falcon_files}
   ${CMAKE_CURRENT_BINARY_DIR}/falcon/config.h)

set(FALCON_SRV_HEADERS
   ${falcon_srv} )

install(FILES ${FALCON_HEADERS} DESTINATION ${FALCON_INC_DIR}/falcon)
install(FILES ${FALCON_SRV_HEADERS} DESTINATION ${FALCON_INC_DIR}/falcon/srv)

set(FALCON_HEADERS ${FALCON_HEADERS} PARENT_SCOPE)
set(FALCON_SRV_HEADERS ${FALCON_SRV_HEADERS} PARENT_SCOPE)