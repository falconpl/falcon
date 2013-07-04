
include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_basic
   embed_basic.cpp
)

add_executable( embed_sync
   embed_sync.cpp
)

add_executable( embed_dyn
   embed_dyn.cpp
)

target_link_libraries( embed_basic 
  falcon_engine
)

target_link_libraries( embed_sync 
  falcon_engine
)

target_link_libraries( embed_dyn 
  falcon_engine
)
