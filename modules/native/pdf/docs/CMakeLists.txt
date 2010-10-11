find_package(faldoc REQUIRED)

file(GLOB_RECURSE scriptExtensions_SRCS "../src/scriptExtensions/*.cpp")
faldoc_module_docs(hpdf
  faldoc.fd.in # input
  ${scriptExtensions_SRCS} # additional depenencies
)

