
PROJECT(embed_004)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_004
   main.cpp  
)

target_link_libraries( embed_004
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS embed_004
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
