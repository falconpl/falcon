/*
   FALCON - The Falcon Programming Language.
   FILE: hash_st.h

   Provides multiple hashing algorithms
   Interface extension functions
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Thu, 25 Mar 2010 02:46:10 +0100

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
// Falcon::VMachine::moduleString( hash_msg ) will
// return the associated string or the internationalized version.
// FAL_STR( hash_msg ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( hash_err_finalized, "Hash already finalized" );
FAL_MODSTR( hash_err_not_membuf_1, "Returned type is not MemBuf with word size 1" );
FAL_MODSTR( hash_err_membuf_length_differs, "Returned MemBuf length is not what bytes() returns, fix this" );
FAL_MODSTR( hash_err_size, "Hash can't have length 0" );
FAL_MODSTR( hash_err_no_overload, "Method not overloaded" );

/* end of hash_st.h */
