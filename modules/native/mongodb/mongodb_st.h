/*
 *  Falcon MongoDB - Strings
 */

#ifndef MONGODB_ST_H
#define MONGODB_ST_H

#include <falcon/message_defs.h>

FAL_MODSTR( _err_nomem, "No memory left" );
FAL_MODSTR( _err_create_conn, "Unable to create a MongoDB connection" );
FAL_MODSTR( _err_connect_bad_arg, "Bad arguments" );
FAL_MODSTR( _err_connect_no_socket, "No socket" );
FAL_MODSTR( _err_connect_fail, "Connection failed" );
FAL_MODSTR( _err_connect_not_master, "Not master" );


#endif // !MONGODB_ST_H
