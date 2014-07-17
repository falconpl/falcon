
PROJECT(embed_000)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( 000
   main.cpp  
)

target_link_libraries( 000
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS 000 
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
