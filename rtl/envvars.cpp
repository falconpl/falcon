/*
   FALCON - The Falcon Programming Language
   FILE: envvars.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer dic 27 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/sys.h>
#include <falcon/vm.h>
#include "falcon_rtl_ext.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC  falcon_getenv( ::Falcon::VMachine *vm )
{
   Item *i_var = vm->param( 0 );
   if ( i_var == 0 || ! i_var->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String retVal;
   if ( Sys::_getEnv( *i_var->asString(), retVal ) )
   {
      vm->retval( retVal ); // will garbage this
   }
   else {
      vm->retnil();
   }
}

FALCON_FUNC  falcon_setenv( ::Falcon::VMachine *vm )
{
   Item *i_var = vm->param( 0 );
   Item *i_value = vm->param( 1 );
   if ( i_var == 0 || ! i_var->isString() || i_value == 0  )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *value;
   String localVal;

   if ( i_value->isString() )
      value = i_value->asString();
   else
   {
      value = &localVal;
      vm->itemToString( *value, i_value );
   }

   if ( ! Sys::_setEnv( *i_var->asString(), *value ) )
   {
      vm->raiseError( 10100, "Environment variable set failed." );
   }
}

FALCON_FUNC  falcon_unsetenv( ::Falcon::VMachine *vm )
{
   Item *i_var = vm->param( 0 );
   if ( i_var == 0 || ! i_var->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Sys::_unsetEnv( *i_var->asString() );
}

}
}


/* end of envvars.cpp */
