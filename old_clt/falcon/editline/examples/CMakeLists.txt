include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(fileman fileman.c)
target_link_libraries(fileman edit)
    
add_executable(test test.c)
target_link_libraries(test edit)

add_executable(maik maik.cpp)
target_link_libraries(maik edit)

add_executable(iohub iohub.cpp)
target_link_libraries(iohub edit)
