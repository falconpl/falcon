/*
   FALCON - The Falcon Programming Language.
   FILE: process_st.h

   Process control module - String table.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 May 2008 17:39:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process control module - String table.
*/

#ifndef FLC_process_ST_H
#define FLC_process_ST_H

#include <falcon/module.h>

FAL_MODSTR( proc_msg_errlist,  "Error while reading the process list" );
FAL_MODSTR( proc_msg_errlist2, "Error while closing the process list" );
FAL_MODSTR( proc_msg_errlist3, "Error while creating the process list");
FAL_MODSTR( proc_msg_allstr, "All the elements in the first parameter must be strings" );
FAL_MODSTR( proc_msg_prccreate, "Can't open the process" );
FAL_MODSTR( proc_msg_waitfail, "Wait failed" );
FAL_MODSTR( proc_msg_termfail, "Terminate failed" );

#endif

/* process_st.h */
