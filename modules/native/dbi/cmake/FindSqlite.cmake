find_path(Sqlite_INCLUDE_DIR sqlite3.h)
find_library(Sqlite_LIBRARY sqlite3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sqlite DEFAULT_MSG Sqlite_LIBRARY Sqlite_INCLUDE_DIR)

set(Sqlite_INCLUDE_DIRS ${Sqlite_INCLUDE_DIR})
set(Sqlite_LIBRARIES ${Sqlite_LIBRARY})
