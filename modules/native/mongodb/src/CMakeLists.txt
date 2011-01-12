##
#   MongoDB Pure C driver
#

include_directories( BEFORE . )

if ( MSVC )
    add_definitions( -DMONGO_USE__INT64 )
else()
    add_definitions( -fPIC )
    if ( MONGO_HAVE_STDINT )
        add_definitions( -DMONGO_HAVE_STDINT )
    else()
        add_definitions( -DMONGO_USE_LONG_LONG_INT )
    endif()
endif()

set( MONGO_SRCS
    bson.c
    md5.c
    mongo.c
    numbers.c )

add_library( mongoc STATIC ${MONGO_SRCS} )
