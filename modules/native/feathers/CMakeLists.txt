####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################
cmake_minimum_required(VERSION 2.6.2)
PROJECT(Falcon_Feathers)

# Find our Find*.cmake files 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

find_package(Falcon
   HINTS ${Falcon_Core_BINARY_DIR}/devtools
         "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
         "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
         "$ENV{FALCON_ACTIVE_TREE}/cmake"
   REQUIRED
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${Falcon_INCLUDE_DIRS}
)

unset(_project_targets)
#
add_subdirectory(bitbuf)
add_subdirectory(cmdline)
add_subdirectory(confparser)

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

#
add_custom_target(Falcon_Feathers)
add_dependencies(Falcon_Feathers ${_project_targets})
unset(_project_targets)

