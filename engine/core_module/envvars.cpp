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
#include "core_module.h"

/*#

   @funset core_environ Environment support
   @brief Functions used to access the process environment variables.

   Environment variables are an handful way to provide system wide
   configuration. Falcon RTL getenv(), setenv() and unsetenv() functions peek and
   manipulates environment variables.

   Variables set with "setenv()" will be available to child processes in case
   they are launched with the utilities in the Process module.

   @begingroup core_syssupport
   @beginset core_environ
*/

namespace Falcon {
namespace core {

/*#
   @function getenv
   @brief Get environment variable value.
   @param varName Environment variable name (as a string)
   @return The value of the environment variable or nil if it is not present.

   This function returns a string containing the value set for the given
   environment variable by the operating system before starting the Falcon process
   or or by a previous call to setenv(). If the given variable name is not
   declared, the function will return nil.

   On some systems (e.g. MS-Windows), setting a variable to an empty string is
   equivalent to unsetting it, so getenv() will never return an empty string. On
   other systems, environment variables may be set to empty strings, that may be
   returned by getenv().
*/

FALCON_FUNC  falcon_getenv( ::Falcon::VMachine *vm )
{
   Item *i_var = vm->param( 0 );
   if ( i_var == 0 || ! i_var->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
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


/*#
   @function setenv
   @brief Set environment variable value.
   @param varName Environment variable name (as a string)
   @param value a value for the given variable.
   @raise IoError on failure.

   This function sets the given value for the given environment variable. The
   varName parameter must be a string, while value may be any Falcon value. If the
   value is not a string, it will be converted
   using the toString() function.

   If the variable was previously set to a different value, its value is changed;
   if it doesn't existed, it is created.

   The function may fail if the system cannot perform the operation; this may
   happen if the space that the system reserves for environment variables is
   exhausted. In this case, the function raises an error.
*/

FALCON_FUNC  falcon_setenv( ::Falcon::VMachine *vm )
{
   Item *i_var = vm->param( 0 );
   Item *i_value = vm->param( 1 );
   if ( i_var == 0 || ! i_var->isString() || i_value == 0  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
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
      throw new IoError( ErrorParam( 1000, __LINE__ ).
         origin( e_orig_runtime ).desc( "Environment variable set failed." ).
         extra( *i_var->asString() ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}


/*#
   @function unsetenv
   @brief Clear environment variable value.
   @param varName Environment variable name (as a string)

   This function removes a given variable setting, causing
   subsequents getenv( varName ) to return nil.
*/

FALCON_FUNC  falcon_unsetenv( ::Falcon::VMachine *vm )
{
   Item *i_var = vm->param( 0 );
   if ( i_var == 0 || ! i_var->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   Sys::_unsetEnv( *i_var->asString() );
}

}
}


/* end of envvars.cpp */
