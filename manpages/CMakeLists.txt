#
# Install all the manual files.
#

cmake_minimum_required(VERSION 2.6)

#Find gzip
FIND_PROGRAM(GZIP_TOOL
             NAMES gzip
             PATHS /bin
                   /usr/bin
                   /usr/local/bin)

IF(NOT GZIP_TOOL)
  MESSAGE(FATAL_ERROR "Unable to find 'gzip' program")
ENDIF(NOT GZIP_TOOL)

# Use default man dir
if (NOT FALCON_MAN_DIR)
   set( FALCON_MAN_DIR "share/man/man1" )
endif()

file(GLOB SRC "*.1" )

FOREACH(man ${SRC})
   get_filename_component( mantarget ${man} NAME )
   
   ADD_CUSTOM_TARGET( ${mantarget} ALL
                     ${GZIP_TOOL} --best -c ${man} > ${mantarget}.gz
                     DEPENDS ${man}
                     COMMENT "Building ${mantarget}.gz")

   INSTALL( FILES ${CMAKE_CURRENT_BINARY_DIR}/${mantarget}.gz
            DESTINATION ${FALCON_MAN_DIR}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ RENAME )
            
   list( APPEND targets ${mantarget}.gz )
ENDFOREACH()

