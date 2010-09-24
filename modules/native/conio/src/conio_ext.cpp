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
   Interface extension functions
*/

#include <falcon/engine.h>
#include "conio_mod.h"
#include "conio_ext.h"
#include "conio_st.h"
#include <conio_srv.h>

extern Falcon::Srv::ConsoleSrv *console_service;

namespace Falcon {
namespace Ext {

// The following is a faldoc block for the function
/*#
   @function initscr
   @brief Initializes the screen.
   @optparam flags Initialization params.
   @raise ConioError on initialization failed.

   @todo describe flags.
   Initializes the screen.
*/

FALCON_FUNC  initscr( ::Falcon::VMachine *vm )
{
   Item *i_flags = vm->param( 0 );

   if( i_flags != 0 && ! i_flags->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra("[N]") ) );
      return;
   }

   // get the flags...
   int flags = i_flags == 0 ? 0 : (int) i_flags->forceInteger();

   //... init things...
   if ( console_service->init() != Srv::ConsoleSrv::e_none )
   {
      vm->raiseModError( new ConioError( ErrorParam( 5000, __LINE__ ).
         desc( FAL_STR( conio_msg_2 ) ) ) );
   }
}

/*#
   @function ttString
   @brief Write teletype a given string.
   @param str The string to be written.

   Writes a string on the screen and advaces the position
   of the logical cursor.
*/
FALCON_FUNC  ttString( ::Falcon::VMachine *vm )
{
   Item *i_str = vm->param( 0 );

   if( i_str == 0 || ! i_str->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("S") ) );
      return;
   }

   AutoCString cstr( * i_str->asString() );
   //... use the string...

   // ... let's return a fake value...
   vm->retval( (int64)cstr.length() );
}

/*#
   @function closescr
   @brief Deinitializes the screen.
   @raise ConioError on deinitialization failed.

   Closes the screen.
*/
FALCON_FUNC  closescr( ::Falcon::VMachine *vm )
{
   // chiudi
   console_service->shutdown();
}


/*#
   @class ConioError
   @brief Error generated because of problems on the console I/O.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   See the Error class in the core module.
*/

/*#
   @init ConioError
   @brief Initializes the process error.
*/
FALCON_FUNC  ConioError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new ConioError );

   ::Falcon::core::Error_init( vm );
}



}
}

/* end of conio_mod.cpp */
