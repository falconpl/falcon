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
#include "curl_mod.h"
#include "curl_ext.h"
#include "curl_st.h"

namespace Falcon {
namespace Ext {

// The following is a faldoc block for the function
/*#
   @function skeleton
   @brief A basic script function.
   @return Zero.

   This function just illustrates how to bind the ineer MOD logic
   with the script. Also, Mod::skeleton(), used by this
   function, is exported through the "service", so it is
   possible to call the MOD logic directly from an embedding
   application, once loaded the module and accessed the service.
*/

FALCON_FUNC  skeleton( ::Falcon::VMachine *vm )
{
   vm->retval( (int64) Mod::skeleton() );
}

/*#
   @function skeletonString
   @brief A function returning a string in the string table.
   @return A message that can be internationalized.

   This function returns a string from the string table
   (see curl_st.h).

   The returned string may be internationalized through
   the standard falcon internationalization system (the
   same available for scripts).

   A real module will want to use this system to produce
   locale-configurable messages (expecially the "extra"
   field of error descriptions).
*/
FALCON_FUNC  skeletonString( ::Falcon::VMachine *vm )
{
   vm->retval( FAL_STR( curl_msg_1 ) );
}

}
}

/* end of curl_mod.cpp */
