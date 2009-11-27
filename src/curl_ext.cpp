/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

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
   cURL library binding for Falcon
   Interface extension functions
*/

#include <falcon/engine.h>
#include <curl/curl.h>

#include "curl_mod.h"
#include "curl_ext.h"
#include "curl_st.h"

namespace Falcon {
namespace Ext {

// The following is a faldoc block for the function
/*#
   @function curl_version
   @brief Returns the version of libcurl
   @return A string containing the description of the used version-.
*/

FALCON_FUNC  curl_version( ::Falcon::VMachine *vm )
{
   vm->retval( new CoreString( ::curl_version() ) );
}

}
}

/* end of curl_mod.cpp */
