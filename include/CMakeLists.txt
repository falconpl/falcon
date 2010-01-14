####################################################################
# The Falcon Programming language
#
# Install service directories for services offered by modules
####################################################################

file(GLOB service_headers "falcon/srv/*.h" )

install(
  FILES ${service_headers}
  DESTINATION ${falcon_INC_DIR}/srv
) 
