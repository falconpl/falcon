# benchlog: obj/test/regexp_benchmark
#      (echo '==BENCHMARK==' `hostname` `date`; \
#        (uname -a; g++ --version; hg identify; file obj/test/regexp_benchmark) | sed 's/^/# /'; \
#        echo; \
#        ./obj/test/regexp_benchmark 'PCRE|RE2') | tee -a benchlog.$$(hostname | sed 's/\..*//')

# TODO (not available with CMake commands or variables)
# * Hostname
# * Date
# * uname -a
# * hg identify
# * file obj/tes/regexp_benchmark
#
message( STATUS "# ==BENCHMARK== " )
message( STATUS "# ${CMAKE_CXX_COMPILER}" )
execute_process( COMMAND regexp_benchmark 'PCRE|RE2'
                   OUTPUT_FILE benchlog
                 )
