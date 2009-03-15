/*
   FALCON - The Falcon Programming Language.
   FILE: threading_st.h

   Threading module - String table.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Jun 2008 20:57:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Threading module - String table.
*/

#include <falcon/message_defs.h>

FAL_MODSTR( th_msg_notrunnable,  "Not runnable" );
FAL_MODSTR( th_msg_running, "Thread is already running" );
FAL_MODSTR( th_msg_errlink, "Failed thread setup (vm link)" );
FAL_MODSTR( th_msg_errstart, "Failed to start the thread" );
FAL_MODSTR( th_msg_notrunning, "Thread not running" );
FAL_MODSTR( th_msg_threadnotterm, "Thread not terminated" );
FAL_MODSTR( th_msg_ejoin, "Unjoinable thread" );
FAL_MODSTR( th_msg_joinwitherr, "Joined thread terminated with error" );

FAL_MODSTR( th_msg_qempty, "Queue is empty" );
FAL_MODSTR( th_msg_errdes, "Error in deserializing an item" );

/* threading_st.h */
