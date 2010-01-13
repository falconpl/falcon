####################################################################
# The Falcon Programming language
#
# Install service directories for services offered by modules
####################################################################

file(GLOB service_headers "falcon/srv/*.h" )

install(
  FILES ${service_headers}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/falcon/srv
) 
