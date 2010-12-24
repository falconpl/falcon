##
#   MongoDB Pure C driver
#

include_directories( BEFORE . )

if ( MSVC )
    add_definitions( -DMONGO_USE__INT64 )
else()
    add_definitions( --std=c99 )
endif()

set( MONGO_SRCS
    bson.c
    md5.c
    mongo.c
    numbers.c )

add_library( mongoc STATIC ${MONGO_SRCS} )
