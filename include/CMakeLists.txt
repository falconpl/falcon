####################################################################
# The Falcon Programming language
#
# Install header files for module development
###################################################################

# As the prefix "falcon" is hardcoded in directives #include <falcon/...>
# we want to have include hierarcy as /usr/bin/include/falcon1.0.0/falcon
# while the symlink include/falcon will point not to falcon1.0.0 but to "falcon1.0.0/falcon"
 
set( TARGET_INCLUDE_DIR "${PROJECT_BINARY_DIR}/${FALCON_INC_DIR}/falcon" )
configure_file( 
  falcon/config.h.in  "${TARGET_INCLUDE_DIR}/config.h"
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
   ${FALCON_INC_DIR}/falcon/config.h)

install(FILES ${FALCON_HEADERS} DESTINATION ${FALCON_INC_DIR}/falcon)
install(FILES ${falcon_classes} DESTINATION ${FALCON_INC_DIR}/falcon/classes)
install(FILES ${falcon_cm} DESTINATION ${FALCON_INC_DIR}/falcon/cm)
install(FILES ${falcon_errors} DESTINATION ${FALCON_INC_DIR}/falcon/errors)
install(FILES ${falcon_parser} DESTINATION ${FALCON_INC_DIR}/falcon/parser)
install(FILES ${falcon_psteps} DESTINATION ${FALCON_INC_DIR}/falcon/psteps)
install(FILES ${falcon_sp} DESTINATION ${FALCON_INC_DIR}/falcon/sp)
install(FILES ${falcon_tc} DESTINATION ${FALCON_INC_DIR}/falcon/tc)

#Create a falcon -> falconX.X.X symlink on install.
if(FALCON_SYMLINK_VERSION AND (UNIX OR APPLE) )
 install(CODE "
    EXECUTE_PROCESS(COMMAND ln -sf \"${FALCON_INC_DIR}/falcon\" falcon
       WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
       )
    ")
endif()

set(FALCON_HEADERS ${FALCON_HEADERS} PARENT_SCOPE)

