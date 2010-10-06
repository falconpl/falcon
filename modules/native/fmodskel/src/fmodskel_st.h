/*
   @MAIN_PRJ@
   FILE: @PROJECT_NAME@_st.h

   @DESCRIPTION@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @AUTHOR@
   Begin: @DATE@

   -------------------------------------------------------------------
   (C) Copyright @YEAR@: @COPYRIGHT@

   @LICENSE@
*/

/** \file
   String table - export internationalizable stings to both C++
                  and Falcon modules.
*/

//WARNING: the missing of usual #ifndef/#define pair
//         is intentional!

// See the contents of this header file for a deeper overall
// explanation of the MODSTR system.
#include <falcon/message_defs.h>

// The first parameter is an unique identifier in your project that
// will be bound to the correct entry in the module string table.
// Falcon::VMachine::moduleString( @PROJECT_NAME@_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( @PROJECT_NAME@_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( @PROJECT_NAME@_msg_1, "An internationalizable message" );

//... add here your messages, and remove or configure the above one

/* end of @PROJECT_NAME@_st.h */
