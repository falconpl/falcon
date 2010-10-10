find_package(faldoc REQUIRED)

file(GLOB_RECURSE scriptExtensions_SRCS "../*.cpp")
faldoc_module_docs(sdl
  faldoc.fd.in # input
  ${scriptExtensions_SRCS} # additional depenencies
  main.fd
)
