/*
   @{fmodskel_MAIN_PRJ}@
   FILE: @{fmodskel_PROJECT_NAME}@_st.h

   @{fmodskel_DESCRIPTION}@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @{fmodskel_AUTHOR}@
   Begin: @{fmodskel_DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{fmodskel_YEAR}@: @{fmodskel_COPYRIGHT}@

   @{fmodskel_LICENSE}@
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
// Falcon::VMachine::moduleString( @{fmodskel_PROJECT_NAME}@_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( @{fmodskel_PROJECT_NAME}@_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( @{fmodskel_PROJECT_NAME}@_msg_1, "An internationalizable message" );

//... add here your messages, and remove or configure the above one

/* end of @{fmodskel_PROJECT_NAME}@_st.h */
