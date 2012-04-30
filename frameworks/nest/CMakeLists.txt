############################################################
# CMake list for nest framework.
#

set( module_dirs
   nest
)

falcon_install_moddirs( ${module_dirs} )
install( DIRECTORY nest/res DESTINATION "${FALCON_MOD_DIR}/nest" )
