/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_st.h

   Database interface - String table.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 09 May 2010 12:26:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Regular expression module - String table.
*/

#include <falcon/message_defs.h>

FAL_MODSTR( dbi_msg_invalid_col, "Column out of range" );
FAL_MODSTR( dbi_msg_driver_not_found, "DBI driver service not found" );
FAL_MODSTR( dbi_msg_nomem, "Not enough memory to perform the operation" );
FAL_MODSTR( dbi_msg_connparams, "Malformed or invalid connection parameter string" );
FAL_MODSTR( dbi_msg_connect, "Connection to database failed" );
FAL_MODSTR( dbi_msg_query, "Database query error" );
FAL_MODSTR( dbi_msg_query_empty, "Query didn't return any result" );


/* dbi_st.h */
