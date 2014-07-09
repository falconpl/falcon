####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################
cmake_minimum_required(VERSION 2.6.2)

Message(STATUS "Bulding Feather modules")

if(NOT FALCON_STATIC_FEATHERS )
	PROJECT(Falcon_Feathers)
endif()

#find_package(Falcon
#   HINTS ${Falcon_Core_BINARY_DIR}/devtools
#         "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
#         "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
#         "$ENV{FALCON_ACTIVE_TREE}/cmake"
#   REQUIRED
#)

set(Falcon_Feathers_project_targets )
#
add_subdirectory(bitbuf)
add_subdirectory(cmdline)
add_subdirectory(confparser)
add_subdirectory(containers)

##add_subdirectory(funcext)
add_subdirectory(hash)
add_subdirectory(json)
add_subdirectory(inet)
add_subdirectory(logging)
add_subdirectory(math)
add_subdirectory(mxml)
add_subdirectory(process)
add_subdirectory(regex)
add_subdirectory(rnd)
add_subdirectory(shmem)
add_subdirectory(sys)
add_subdirectory(vm)
#add_subdirectory(zlib)

# Virtual file system module.
add_subdirectory(vfs)

if(NOT FALCON_STATIC_FEATHERS )
	add_custom_target(Falcon_Feathers)
	include_directories( "${CMAKE_BINARY_DIR}/${FALCON_INC_DIR}" )
	add_dependencies(Falcon_Feathers "${Falcon_Feathers_project_targets}")
	unset(Falcon_Feathers_project_targets)
endif()
