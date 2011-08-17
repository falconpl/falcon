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

/*#
   @beginmodule core
*/

namespace Falcon {
namespace core {

/*#
   @method name Function
   @brief Gets the symbolic name of the given function.
   @return A string containing the function name

   This is useful if the function symbol or has been
   re-assigned to temporary variables, or if it is applied
   to the @b fself keyword.
*/
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



static void s_caller_internal( VMachine* vm, bool mode )
{
   uint32 level;
   Item *i_level = vm->param(0);

   if( i_level != 0 )
   {
      if( i_level->isOrdinal() || i_level->isNil() )
      {
         int64 i64level =  i_level->forceInteger();
         if( i64level < 0 )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin(e_orig_runtime)
               .extra( ">=0" ) );
         }

         level = (uint32)i64level+1;
      }
      else
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "[N]" ) );
      }
   }
   else {
      level = 1;
   }

   if( mode )
   {
      Item caller;
      if ( vm->getCallerItem( caller, level ) )
      {
         vm->retval( caller );
         return;
      }
   }
   else
   {
      const Symbol *sym;
      uint32 line;
      uint32 pc;
      if( vm->getTraceStep( level, sym, line, pc ) )
      {
         CoreArray* arr = new CoreArray( 3 );
         arr->append( sym->name() );
         arr->append( sym->module()->name() );
         arr->append( sym->module()->path() );
         arr->append( (int64) line );
         arr->append( (int64) pc );
         vm->retval( arr );
         return;
      }
   }

   // we're not called.
   vm->retnil();
}


/*#
   @method caller Function
   @brief Gets the direct caller or one of the calling ancestors.
   @optparam level Caller level (starting from zero, the default).
   @return The item having performed the nth call.

   This function returns the n-th caller (zero based) that caused
   this function to be called. It may be a function, a method
   or another callable item from which the call has originated.

   @note The method can also be called statically on the Function metaclass.
*/

FALCON_FUNC  Function_caller ( ::Falcon::VMachine *vm )
{
   s_caller_internal( vm, true );
}

/*#
   @method trace Function
   @brief Gets a trace step in the call stack.
   @optparam level Caller level (starting from zero, the default).
   @return An array containing the data relative to the given trace level.

   The returned data is organized as follows:
   @code
   [ 'symbol name', 'module name', 'module path', line_in_module, PC_in_vm]
   @endcode

   @note The method can also be called statically on the Function metaclass.
*/

FALCON_FUNC  Function_trace ( ::Falcon::VMachine *vm )
{
   s_caller_internal( vm, false );
}

/*#
   @method stack Function
   @brief Gets a string representation of the call stack.
   @return A string containing all the stack trace up to this point.

   The returned string looks like the stack trace printed by errors,
   but is referred to the call stack up to the function from which
   this method is called.

   @note The method can also be called statically on the Function metaclass.
*/

FALCON_FUNC  Function_stack ( ::Falcon::VMachine *vm )
{
   const Symbol *sym;
   uint32 line;
   uint32 pc;
   int level = 0;

   CoreString& target = *(new CoreString());

   while( vm->getTraceStep( level, sym, line, pc ) )
   {
      if ( target.size() )
         target += "\n";

      const Module* mod = sym->module();
      if ( mod->path().size() )
      {
         target += "\"" + mod->path() + "\" ";
      }

      target += mod->name() + "." + sym->name() + ":";
      target.writeNumber( (int64) line );
      target += "(PC:";
      switch( pc )
      {
         case VMachine::i_pc_call_external: target += "ext"; break;
         case VMachine::i_pc_call_external_return: target += "ext.r"; break;
         case VMachine::i_pc_redo_request: target += "redo"; break;
         case VMachine::i_pc_call_external_ctor: target += "ext.c"; break;
         case VMachine::i_pc_call_external_ctor_return: target += "ext.cr"; break;
         default:
            target.writeNumber( (int64) pc );
      }

      target += ")";

      ++level;
   }

   vm->retval( &target );
}

}
}

/* end of functional_ext.cpp */
