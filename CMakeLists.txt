PROJECT(Falcon_Core)

#Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")

ADD_SUBDIRECTORY(engine)

SET( CONFIG_LIST
   include/falcon/config.h )

FALCON_CONFIG( CONFIG_LIST )


