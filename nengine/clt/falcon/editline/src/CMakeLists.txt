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


set(libedit_SOURCES 
  chared.c
  chartype.c
  el.c
  eln.c
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
  wcsdup.c
  tokenizer.c 
  history.c 
  filecomplete.c 
   
  vi.c
  emacs.c
  common.c
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