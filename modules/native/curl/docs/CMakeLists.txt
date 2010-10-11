find_package(faldoc REQUIRED)

file(GLOB_RECURSE scriptExtensions_SRCS "../src/*.cpp")
faldoc_module_docs(curl
  faldoc.fd.in # input
  ${scriptExtensions_SRCS} # additional depenencies
)
