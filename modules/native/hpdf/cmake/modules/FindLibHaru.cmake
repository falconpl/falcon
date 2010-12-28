find_path(LibHaru_INCLUE_DIR hpdf.h)
find_library(LibHaru_LIBRARY hpdf)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibHaru DEFAULT_MSG LibHaru_LIBRARY LibHaru_INCLUE_DIR)

set(LibHaru_INCLUDE_DIRS ${LibHaru_INCLUE_DIR})
set(LibHaru_LIBRARIES ${LibHaru_LIBRARY})