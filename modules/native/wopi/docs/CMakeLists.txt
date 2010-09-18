##########################################################
# Find faldoc in falcon application directory

find_program(
   Falcon_faldoc_EXECUTABLE faldoc.sh  
)

if(NOT Falcon_faldoc_EXECUTABLE )
   find_program( Falcon_faldoc_EXECUTABLE faldoc
      HINT "/usr/bin"
      HINT "/usr/local/bin"  
   )
   
   if( NOT Falcon_faldoc_EXECUTABLE )
      message(FATAL_ERROR "faldoc not found; 
         please specify it via the 
         -DFalcon_faldoc_EXECUTABLE=<path to launch shell>
         setting" 
         )
   endif()
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wopi-docs-1.0/index.html
  COMMAND ${Falcon_faldoc_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
) 

add_custom_target(create_docs ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/wopi-docs-1.0/index.html)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/wopi-docs-1.0/ DESTINATION share/doc/wopi-1.0/html)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/faldoc.fd.in
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
  @ONLY
)
