/*
   FALCON - The Falcon Programming Language.
   FILE: socket_st.h

   Socket module - String table.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 May 2008 17:39:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Socket module - String table.
*/

#ifndef FLC_SOCKET_ST_H
#define FLC_SOCKET_ST_H

#include <falcon/module.h>

FAL_MODSTR( sk_msg_generic, "Generic network error" );
FAL_MODSTR( sk_msg_errresolv, "System error in resolving address" );
FAL_MODSTR( sk_msg_errcreate, "Socket creation failed" );
FAL_MODSTR( sk_msg_errconnect, "Error during connection" );
FAL_MODSTR( sk_msg_errsend, "Network error while sending data" );
FAL_MODSTR( sk_msg_stringnospace, "Passed String must have space" );
FAL_MODSTR( sk_msg_firstnostring, "Given a size, the first parameter must be a string" );
FAL_MODSTR( sk_msg_lesszero, "size less than 0" );
FAL_MODSTR( sk_msg_errrecv, "Network error while receiving data" );
FAL_MODSTR( sk_msg_errclose, "Network error while closing socket" );
FAL_MODSTR( sk_msg_errbind, "Can't bind socket to address" );
FAL_MODSTR( sk_msg_erraccept, "Error while accepting connections" );

#endif

/* socket_st.h */
