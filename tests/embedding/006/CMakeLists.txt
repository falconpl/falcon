
PROJECT(embed_006)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_006
   main.cpp  
)

target_link_libraries( embed_006
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS embed_006
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
