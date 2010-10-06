############################################################
# CMake list for modules.
#

set( module_dirs
   parser
   net
   web
   qrcode
)

falcon_install_moddirs( ${module_dirs} )
