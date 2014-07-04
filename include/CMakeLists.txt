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
file( GLOB falcon_classes falcon/classes/*.h)
file( GLOB falcon_cm falcon/cm/*.h)
file( GLOB falcon_errors falcon/errors/*.h)
file( GLOB falcon_parser falcon/parser/*.h)
file( GLOB falcon_psteps falcon/psteps/*.h)
file( GLOB falcon_sp falcon/sp/*.h)
file( GLOB falcon_tc falcon/tc/*.h)

set(FALCON_HEADERS
   ${falcon_files}
   ${CMAKE_CURRENT_BINARY_DIR}/falcon/config.h)

set(FALCON_SRV_HEADERS
   ${falcon_srv} )

install(FILES ${FALCON_HEADERS} DESTINATION ${FALCON_INC_DIR}/falcon)
install(FILES ${falcon_classes} DESTINATION ${FALCON_INC_DIR}/falcon/classes)
install(FILES ${falcon_cm} DESTINATION ${FALCON_INC_DIR}/falcon/cm)
install(FILES ${falcon_errors} DESTINATION ${FALCON_INC_DIR}/falcon/errors)
install(FILES ${falcon_parser} DESTINATION ${FALCON_INC_DIR}/falcon/parser)
install(FILES ${falcon_psteps} DESTINATION ${FALCON_INC_DIR}/falcon/psteps)
install(FILES ${falcon_sp} DESTINATION ${FALCON_INC_DIR}/falcon/sp)
install(FILES ${falcon_tc} DESTINATION ${FALCON_INC_DIR}/falcon/tc)

set(FALCON_HEADERS ${FALCON_HEADERS} PARENT_SCOPE)
set(FALCON_SRV_HEADERS ${FALCON_SRV_HEADERS} PARENT_SCOPE)

