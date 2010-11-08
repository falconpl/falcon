
if(NOT DEFINED CURL_ROOT)
  if(NOT "$ENV{CURL_ROOT}" STREQUAL "")
    set(CURL_ROOT "$ENV{CURL_ROOT}")
  else()
    set(CURL_ROOT /usr)
  endif()
endif()

find_path(CURL_INCLUDE_DIR 
  NAMES curl/curl.h
  PATHS
    ${CURL_ROOT}/include
)

# Look for the library.
find_library(CURL_LIBRARY 
  NAMES 
    curl
    # Windows MSVC prebuilts:
    curllib
    libcurl_imp
    curllib_static
  PATHS
    ${CURL_ROOT}/lib
    ${CURL_ROOT}/lib64
    ${CURL_ROOT}/bin # dll
)

if(CURL_INCLUDE_DIR AND EXISTS "${CURL_INCLUDE_DIR}/curl/curlver.h")
  file(READ "${CURL_INCLUDE_DIR}/curl/curlver.h" _curlver_h)
  string(REGEX REPLACE ".*#define LIBCURL_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" 
    CURL_VERSION_MAJOR "${_curlver_h}")
  string(REGEX REPLACE ".*#define LIBCURL_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" 
    CURL_VERSION_MINOR "${_curlver_h}")
  string(REGEX REPLACE ".*#define LIBCURL_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" 
    CURL_VERSION_PATCH "${_curlver_h}")
  set(CURL_VERSION "${CURL_VERSION_MAJOR}.${CURL_VERSION_MINOR}.${CURL_VERSION_PATCH}")
  set(CURL_VERSION_STRING "${CURL_VERSION}")
endif()

if(CURL_FIND_VERSION)
  set(CURL_VERSION_FOUND false)
  if(CURL_FIND_VERSION_EXACT)
    if(CURL_FIND_VERSION VERSION_EQUAL "${CURL_VERSION}")
      set(CURL_VERSION_FOUND true)
    else()
      message(STATUS "Didn't find the exact version ${CURL_FIND_VERSION}, but ${CURL_VERSION}")
    endif()
  else()
    if(CURL_FIND_VERSION VERSION_LESS "${CURL_VERSION}" OR CURL_FIND_VERSION VERSION_EQUAL "${CURL_VERSION}")
      set(CURL_VERSION_FOUND true)
    else()
      message(STATUS "Didn't find a sufficient version to meet ${CURL_FIND_VERSION}, but ${CURL_VERSION}")
    endif()
  endif()
else()
  set(CURL_VERSION_FOUND true)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CURL DEFAULT_MSG CURL_LIBRARY CURL_INCLUDE_DIR CURL_VERSION_FOUND)

set(CURL_LIBRARIES ${CURL_LIBRARY})
set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIR})
