
PROJECT(embed_003)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_003
   main.cpp  
)

target_link_libraries( embed_003
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS embed_003
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
