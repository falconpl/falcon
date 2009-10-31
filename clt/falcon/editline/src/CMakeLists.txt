include_directories(
  ${PROJECT_BINARY_DIR} # config.h
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CURSES_INCLUDE_DIR}
)

set(AHDR 
  vi.h 
  emacs.h 
  common.h
) 

set(ASRC 
  ${CMAKE_CURRENT_SOURCE_DIR}/vi.c 
  ${CMAKE_CURRENT_SOURCE_DIR}/emacs.c 
  ${CMAKE_CURRENT_SOURCE_DIR}/common.c
)


add_custom_command(
  OUTPUT ${AHDR}
  DEPENDS vi.c emacs.c common.c
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -h ${CMAKE_CURRENT_SOURCE_DIR}/vi.c > vi.h
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -h ${CMAKE_CURRENT_SOURCE_DIR}/emacs.c > emacs.h
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -h ${CMAKE_CURRENT_SOURCE_DIR}/common.c > common.h
)
add_custom_command(
  OUTPUT fcns.h fcns.c help.h help.c 
  DEPENDS ${AHDR}
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -fh ${AHDR} > fcns.h
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -bh ${ASRC} > help.h
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -fc ${AHDR} > fcns.c
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -bc ${ASRC} > help.c
)

add_custom_target(makelist_target DEPENDS 
  ${AHDR} 
  help.h 
  fcns.h
)
set(libedit_SOURCES 
  chared.c 
  el.c 
  hist.c 
  key.c 
  map.c
  parse.c 
  prompt.c 
  read.c 
  refresh.c 
  search.c
  sig.c 
  term.c 
  tty.c 
  fgetln.c 
  strlcat.c
  strlcpy.c 
  unvis.c 
  vis.c 
  tokenizer.c 
  history.c 
  filecomplete.c 
  readline.c   
)

add_library(edit STATIC 
  ${libedit_SOURCES}
  ${ASRC}
  fcns.c
  help.c
)
add_dependencies(edit makelist_target)
target_link_libraries(edit 
  ${CURSES_LIBRARIES}
)