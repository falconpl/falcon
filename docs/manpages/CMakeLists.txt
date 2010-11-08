#
# Install all the manual files.
#

find_program(GZIP_TOOL
  NAMES gzip
  PATHS
    /bin
    /usr/bin
    /usr/local/bin
)

if(NOT GZIP_TOOL)
  message(FATAL_ERROR "Unable to find 'gzip' program")
endif(NOT GZIP_TOOL)

# Use default man dir
if (NOT FALCON_MAN_DIR)
   set( FALCON_MAN_DIR "share/man/man1" )
endif()

file(GLOB SRC "*.1" )

foreach(man ${SRC})
   get_filename_component( mantarget ${man} NAME )

   add_custom_command(
     OUTPUT ${mantarget}.gz
     COMMAND ${GZIP_TOOL} --best -c ${man} > ${mantarget}.gz
     DEPENDS ${man}
     COMMENT "Building ${mantarget}.gz"
   )
   add_custom_target( ${mantarget} ALL
     DEPENDS ${mantarget}.gz
   )

   install(
     FILES ${CMAKE_CURRENT_BINARY_DIR}/${mantarget}.gz
     DESTINATION ${FALCON_MAN_DIR}
     PERMISSIONS OWNER_READ GROUP_READ WORLD_READ RENAME
   )

   list( APPEND targets ${mantarget}.gz )
endforeach()
