#/# editline for falcon

add_definitions( -D_GNU_SOURCE )

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
  OUTPUT fcns.h help.h func.h
  DEPENDS ${AHDR}
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -fh ${AHDR} > fcns.h
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -bh ${ASRC} > help.h
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/makelist -fc ${AHDR} > func.h
)

add_custom_target(makelist_target DEPENDS
  ${AHDR}
  help.h
  fcns.h
  func.h
)
set(libedit_SOURCES
  chared.c
  el.c
  eln.c
  hist.c
  keymacro.c
  map.c
  chartype.c
  parse.c
  prompt.c
  read.c
  refresh.c
  search.c
  sig.c
  terminal.c
  tty.c
  reallocarr.c
  wcsdup.c
  tokenizer.c
  tokenizern.c
  history.c
  historyn.c
  filecomplete.c
  readline.c
  literal.c
  strlcpy.c
  strlcat.c
  vis.c
  unvis.c
)

add_library(edit STATIC
  ${libedit_SOURCES}
  ${ASRC}
)
add_dependencies(edit makelist_target)
target_link_libraries(edit
  ${CURSES_LIBRARIES}
)

# vi: ts=2 sw=2 et
