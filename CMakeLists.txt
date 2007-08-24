PROJECT(Falcon_Core)

#Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")

ADD_SUBDIRECTORY(engine)
INCLUDE_DIRECTORIES(include)
