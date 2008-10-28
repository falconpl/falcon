/*
   The Falcon Programming Language
   FILE: dynlib_ext.cpp

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
   Direct dynamic library interface for Falcon
   Interface extension functions - header file
*/

#ifndef dynlib_ext_H
#define dynlib_ext_H

#include <falcon/module.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC  DynLib_init( ::Falcon::VMachine *vm );
FALCON_FUNC  DynLib_get( ::Falcon::VMachine *vm );
FALCON_FUNC  DynLib_unload( ::Falcon::VMachine *vm );

}
}

#endif

/* end of dynlib_ext.h */
