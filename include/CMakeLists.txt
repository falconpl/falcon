####################################################################
# The Falcon Programming language
#
# Install service directories for services offered by modules
####################################################################

file(GLOB service_headers "falcon/srv/*.h" )

install(
  FILES ${service_headers}
  DESTINATION ${Falcon_INCLUDE_DIR}/falcon/srv
) 
