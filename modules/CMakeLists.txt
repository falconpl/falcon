############################################################
# CMake list for modules.
#

set( module_dirs
   parser
   net
   web
   )

message( "Installing top modules in ${CMAKE_CURRENT_SOURCE_DIR}" )
file( GLOB modules "*.fal" )
foreach( single_fal ${modules} )
   install(
      FILES ${single_fal}
      DESTINATION ${FALCON_MOD_DIR}
   )
endforeach()

foreach(item ${module_dirs} )
   message( "Installing falcon modules in ${item}" )
   file( GLOB_RECURSE files "${item}" "*.fal" "*.ftd" )
   foreach( single_fal ${files} )
      file( RELATIVE_PATH single_fal_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${single_fal}")
      get_filename_component( path_of_fal "${single_fal_relative}"  PATH)

      #Create installation files from in files
      install(
         FILES "${single_fal}"
         DESTINATION "${FALCON_MOD_DIR}/${path_of_fal}"
      )
   endforeach()
    

endforeach()


