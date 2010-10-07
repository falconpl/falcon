##################################################
# Falcon Programming Language
#
# CLT - Command Line Tools meta cmake file
##################################################

ADD_SUBDIRECTORY(faltest)
ADD_SUBDIRECTORY(faldisass)
ADD_SUBDIRECTORY(falrun)
ADD_SUBDIRECTORY(falcon)
ADD_SUBDIRECTORY(falpack)

configure_file(
  executable_wrapper.cmake.in
  ${CMAKE_BINARY_DIR}/${FALCON_BIN_DIR}/executable_wrapper.cmake
  @ONLY
)
