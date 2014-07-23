
PROJECT(embed_007)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_007
   main.cpp  
)

target_link_libraries( embed_007
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS embed_007
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
