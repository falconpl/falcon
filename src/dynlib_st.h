/*
   The Falcon Programming Language
   FILE: dynlib_st.h

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

      Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

      http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

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
// Falcon::VMachine::moduleString( dynlib_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( dynlib_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( dynlib_msg_1, "An internationalizable message" );

//... add here your messages, and remove or configure the above one

/* end of dynlib_st.h */
