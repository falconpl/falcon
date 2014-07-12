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

#ifndef FALCON_MODULES_CURL_ST_H
#define FALCON_MODULES_CURL_ST_H

#define curl_err_desc "CURL error code:"
#define curl_err_init "Error during intialization"
#define curl_err_exec "Error during transfer"
#define curl_err_resources "Not enough resources to complete the operation"
#define curl_err_pm "Curl handle already closed"
#define curl_err_setopt "Type of parameter incompatible for this option"
#define curl_err_unkopt "Unknown option for setOption"
#define curl_err_getinfo "Error while reading required information."
#define curl_err_easy_already_in "Handle already added"
#define curl_err_easy_not_in "Handle currently not present"
#define curl_err_multi_error "Error in CURL multiple operation"

#endif

/* end of curl_st.h */
