/*
   FALCON - The Falcon Programming Language.
   FILE: MP_st.h

   Multi-Precision Math support
   Interface extension functions
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Fri, 12 Mar 2010 15:58:42 +0000

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

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
// Falcon::VMachine::moduleString( MP_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( MP_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( MP_msg_1, "An internationalizable message" );

//... add here your messages, and remove or configure the above one

/* end of MP_st.h */
