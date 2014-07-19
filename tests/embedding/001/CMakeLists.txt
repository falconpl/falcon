
PROJECT(embed_001)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( 001
   main.cpp  
)

target_link_libraries( 001
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS 001
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
