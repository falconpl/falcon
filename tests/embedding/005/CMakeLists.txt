
PROJECT(embed_005)

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)

add_executable( embed_005
   main.cpp  
)

target_link_libraries( embed_005
  falcon_engine
)

if(FALCON_INSTALL_BINTESTS)
   install(TARGETS embed_005
            DESTINATION "${FALCON_BIN_DIR}/tests")
endif()
