/*
   FALCON - The Falcon Programming Language.
   FILE: MP_ext.cpp

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
   Multi-Precision Math support
   Interface extension functions - header file
*/

#ifndef MP_ext_H
#define MP_ext_H

#include <falcon/module.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC  MPZ_init( ::Falcon::VMachine *vm );
FALCON_FUNC  MPZ_add( ::Falcon::VMachine *vm );
FALCON_FUNC  MPZ_sub( ::Falcon::VMachine *vm );
FALCON_FUNC  MPZ_toString( ::Falcon::VMachine *vm );

}
}

#endif

/* end of MP_ext.h */
