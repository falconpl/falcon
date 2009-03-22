/*
   FALCON - The Falcon Programming Language.
   FILE: function_ext.cpp

   Methods of the function class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Mar 2009 00:12:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/corefunc.h>

namespace Falcon {
namespace core {

FALCON_FUNC  Function_name ( ::Falcon::VMachine *vm )
{
   if ( vm->self().isFunction() )
   {
      vm->retval( vm->self().asFunction()->symbol()->name() );
   }
   else {
      vm->retnil();
   }
}


FALCON_FUNC  Function_caller ( ::Falcon::VMachine *vm )
{
   // static method.
   Item caller;
   uint32 level;
   Item *i_level = vm->param(0);

   if( i_level != 0 )
   {
      if( i_level->isOrdinal() )
      {
         int64 i64level =  i_level->forceInteger();
         if( i64level < 0 )
         {
            vm->raiseRTError( new ParamError( ErrorParam( e_param_range, __LINE__ ).
               extra( "N" ) ) );
            return;
         }

         level = (uint32)i64level+1;
      }
      else
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) );
         return;
      }
   }
   else {
      level = 1;
   }

   if ( vm->getCallerItem( caller, level ) )
   {
      vm->retval( caller );
   }
   else
   {
      // we're not called.
      vm->retnil();
   }
}

}
}

/* end of functional_ext.cpp */
