
PROJECT(embed_002)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_002
   main.cpp  
)

target_link_libraries( embed_002
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS embed_002
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
