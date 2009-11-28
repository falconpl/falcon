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
   Interface extension functions - header file
*/

#ifndef curl_ext_H
#define curl_ext_H

#include <falcon/module.h>

#ifndef FALCON_ERROR_CURL_BASE
#define FALCON_ERROR_CURL_BASE            2350
#endif

#define FALCON_ERROR_CURL_INIT            (FALCON_ERROR_CURL_BASE+0)
#define FALCON_ERROR_CURL_EXEC            (FALCON_ERROR_CURL_BASE+1)

namespace Falcon {
namespace Ext {

FALCON_FUNC  curl_version( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_exec( ::Falcon::VMachine *vm );


FALCON_FUNC  CurlError_init ( ::Falcon::VMachine *vm );
}
}

#endif

/* end of curl_ext.h */
