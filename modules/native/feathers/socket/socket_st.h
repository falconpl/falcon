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

#include <falcon/message_defs.h>

FAL_MODSTR( sk_msg_generic, "Generic network error" );
FAL_MODSTR( sk_msg_errresolv, "System error in resolving address" );
FAL_MODSTR( sk_msg_errcreate, "Socket creation failed" );
FAL_MODSTR( sk_msg_errconnect, "Error during connection" );
FAL_MODSTR( sk_msg_errsend, "Network error while sending data" );
FAL_MODSTR( sk_msg_zeroread, "Required to perform a read of zero or less data." );
FAL_MODSTR( sk_msg_nobufspace, "Not enough space left in the MemBuf to complete the operation." );
FAL_MODSTR( sk_msg_errrecv, "Network error while receiving data" );
FAL_MODSTR( sk_msg_errclose, "Network error while closing socket" );
FAL_MODSTR( sk_msg_errbind, "Can't bind socket to address" );
FAL_MODSTR( sk_msg_erraccept, "Error while accepting connections" );

/* socket_st.h */
