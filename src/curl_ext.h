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
#define FALCON_ERROR_CURL_PM              (FALCON_ERROR_CURL_BASE+2)
#define FALCON_ERROR_CURL_SETOPT          (FALCON_ERROR_CURL_BASE+3)
#define FALCON_ERROR_CURL_GETINFO         (FALCON_ERROR_CURL_BASE+4)
#define FALCON_ERROR_CURL_HISIN           (FALCON_ERROR_CURL_BASE+5)
#define FALCON_ERROR_CURL_HNOIN           (FALCON_ERROR_CURL_BASE+6)
#define FALCON_ERROR_CURL_MULTI           (FALCON_ERROR_CURL_BASE+7)

namespace Falcon {
namespace Ext {

FALCON_FUNC  curl_dload( ::Falcon::VMachine *vm );
FALCON_FUNC  curl_version( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_exec( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_setOutConsole( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_setOutString( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_setOutStream( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_setOutCallback( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_getInfo( ::Falcon::VMachine *vm );

FALCON_FUNC  Handle_setInCallback( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_setInStream( ::Falcon::VMachine *vm );

FALCON_FUNC  Handle_postData( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_setOption( ::Falcon::VMachine *vm );
//FALCON_FUNC  Handle_setOutMessage( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_cleanup( ::Falcon::VMachine *vm );
FALCON_FUNC  Handle_getData( ::Falcon::VMachine *vm );

FALCON_FUNC  Multi_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Multi_add( ::Falcon::VMachine *vm );
FALCON_FUNC  Multi_remove( ::Falcon::VMachine *vm );
FALCON_FUNC  Multi_perform( ::Falcon::VMachine *vm );


FALCON_FUNC  CurlError_init ( ::Falcon::VMachine *vm );
}
}

#endif

/* end of curl_ext.h */
