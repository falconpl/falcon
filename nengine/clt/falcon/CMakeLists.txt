##################################################
# Falcon Programming Language
#  -- Falcon command line tool
##################################################

set( SRC
    falcon.cpp
)

add_executable( falcon 
    ${SRC}
)


target_link_libraries( falcon falcon_engine )