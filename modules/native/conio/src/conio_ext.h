/*
   FALCON - The Falcon Programming Language.
   FILE: conio_ext.cpp

   Basic Console I/O support
   Interface extension functions
   -------------------------------------------------------------------
   Author: Unknown author
   Begin: Thu, 05 Sep 2008 20:12:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: The above AUTHOR

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
   Basic Console I/O support
   Interface extension functions - header file
*/

#ifndef conio_ext_H
#define conio_ext_H

#include <falcon/module.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC  initscr( ::Falcon::VMachine *vm );
FALCON_FUNC  ttString( ::Falcon::VMachine *vm );
FALCON_FUNC  closescr( ::Falcon::VMachine *vm );


class ConioError: public ::Falcon::Error
{
public:
   ConioError():
      Error( "ConioError" )
   {}

   ConioError( const ErrorParam &params  ):
      Error( "ConioError", params )
      {}
};

FALCON_FUNC  ConioError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of conio_ext.h */
