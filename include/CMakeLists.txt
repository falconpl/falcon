#Install service directories for services offered by modules
FILE( GLOB service_headers "falcon/srv/*.h" )

INSTALL( FILES ${service_headers}
   DESTINATION "${FALCON_INSTALL_TREE}/include/falcon/srv")
