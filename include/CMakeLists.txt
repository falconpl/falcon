#
# Just install include files so that every part of the project can use it.
#

FILE( GLOB fasm_files "fasm/*.h" )
FILE( GLOB falcon_files "falcon/*.h" )

INSTALL( FILES ${fasm_files}
DESTINATION $ENV{FALCON_BUILD_TREE}/include/fasm)

INSTALL( FILES ${falcon_files}
DESTINATION $ENV{FALCON_BUILD_TREE}/include/falcon)
