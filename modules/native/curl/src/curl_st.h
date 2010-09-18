/*
   FALCON - The Falcon Programming Language.
   FILE: curl_st.h

   cURL library binding for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 27 Nov 2009 16:31:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

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
// Falcon::VMachine::moduleString( curl_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( curl_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( curl_err_desc, "CURL error code:" );
FAL_MODSTR( curl_err_init, "Error during intialization" );
FAL_MODSTR( curl_err_exec, "Error during transfer" );
FAL_MODSTR( curl_err_resources, "Not enough resources to complete the operation" );
FAL_MODSTR( curl_err_pm, "Curl handle already closed" );
FAL_MODSTR( curl_err_setopt, "Type of parameter incompatible for this option" );
FAL_MODSTR( curl_err_unkopt, "Unknown option for setOption" );
FAL_MODSTR( curl_err_getinfo, "Error while reading required information." );
FAL_MODSTR( curl_err_easy_already_in, "Handle already added" );
FAL_MODSTR( curl_err_easy_not_in, "Handle currently not present" );
FAL_MODSTR( curl_err_multi_error, "Error in CURL multiple operation" );


//... add here your messages, and remove or configure the above one

/* end of curl_st.h */
