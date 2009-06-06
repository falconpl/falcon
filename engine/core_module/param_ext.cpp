/*
   FALCON - The Falcon Programming Language.
   FILE: param_ext.cpp

   Variable parameter management support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 01:54:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"

namespace Falcon {
namespace core {

/*#
   @funset varparams_support Variable parameters support
   @brief Functions giving informations on variable parameters.

   Falcon supports variable parameter calling; a function or method may
   access the items that have been used in the parameter call by counting
   them and accessing them one by one.

   Parameter passed by reference may be modified with the appropriate function.

   This functions may be used whether the calling function provides a list of
   formal parameters or not. The first formal parameter will be treated as the
   variable parameter number zero, and the parameter count may be the same as,
   more than or less than the number of formal parameters.
   So, part of the parameters may be accessible via parameter names,
   and the others may be accessed with the functions in this group.
*/

/*#
   @function paramCount
   @return The parameter count.
   @inset varparams_support
   @brief Returns number of parameter that have been passed to the current function or method.

   The return value is the minimum value between the formal parameters declared
   for the current function and the number of actual parameters the caller passed.
   Formal parameters which are declared in the function header, but for which the
   caller didn't provide actual parameters, are filled with nil.
*/

FALCON_FUNC  paramCount ( ::Falcon::VMachine *vm )
{
   // temporarily save the call environment.
   if ( vm->stackBase() == 0 ) {
      vm->raiseRTError( new GenericError( ErrorParam( e_stackuf ) ) );
   }
   else {
      StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
      if( thisFrame->m_stack_base == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_range ) ) );
         return;
      }

      StackFrame *prevFrame = (StackFrame *) &vm->stackItem( thisFrame->m_stack_base - VM_FRAME_SPACE );
      vm->retval( (int64) prevFrame->m_param_count );
   }
}

/*#
   @function parameter
   @brief Gets the Nth parameter
   @inset varparams_support
   @param pnum The ordinal number of the paremeter, zero based
   @return The nth paramter (zero based) or NIL if the parameter is not given.
   @raise AccessError if @b pnum is out of range.

   This function returns the required parameter, being the first one passed to
   the function indicated as 0, the second as 1 and so on. Both formally declared
   parameters and optional parameters can be accessed this way.

   If the given parameter number cannot be accessed, a AccessError is raised.

   @note This function used to be called "paramNumber", and has been renamed in
      version 0.8.10. The function is still aliased throught the old function name
      for compatibility reason, but its usage is deprecated. Use @b parameter instead.
*/

FALCON_FUNC  _parameter ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( number == 0 || ! number->isOrdinal() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_range ).extra( "(N)" ) ) );
      return;
   }

   if ( vm->stackBase() == 0 )
   {
      vm->raiseRTError( new GenericError( ErrorParam( e_stackuf ) ) );
   }
   else {
      uint32 val = (uint32) number->forceInteger();

      StackFrame *thisFrame = (StackFrame *) vm->currentStack().at( vm->stackBase() - VM_FRAME_SPACE );
      uint32 oldbase = thisFrame->m_stack_base;
      if( oldbase == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_range ) ) );
         return;
      }

      // ...but we want the parameter count of our caller.
      StackFrame *prevFrame = (StackFrame *) vm->currentStack().at( oldbase - VM_FRAME_SPACE );
      // ...while the parameters are below our frame's base.

      if( val >= 0 && val < prevFrame->m_param_count )
      {
         val = oldbase - prevFrame->m_param_count - VM_FRAME_SPACE + val;
         vm->retval( *vm->stackItem( val ).dereference() );
      }
      else {
         vm->retnil();
      }
   }
}

/*#
   @function paramIsRef
   @inset varparams_support
   @brief Checks whether the nth parameter has been passed by reference or not.
   @param number The paramter that must be checked (zero based)
   @return true if the parameter has been passed by reference, false otherwise.
   @raise AccessError if @b number is out of range.

   Both assigning a value to a certain parameter and using the paramSet()
   function will change locally the value of the parameter, b
   ut this value won't be reflected in the actual parameter that was used to
   call the function, unless the parameter was explicitly passed by reference.
   In some contexts, it may be useful to know if this is the case.

   If the given parameter number cannot be accessed, a AccessError is raised.
*/

FALCON_FUNC  paramIsRef ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( number == 0 || ! number->isOrdinal() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_range ).extra( "N" ) ) );
      return;
   }

   if ( vm->stackBase() == 0 )
   {
      vm->raiseRTError( new GenericError( ErrorParam( e_stackuf ) ) );
   }
   else
   {
      uint32 val = (uint32) number->forceInteger();

      StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
      uint32 oldbase = thisFrame->m_stack_base;
      if( oldbase == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_stackuf ) ) );
         return;
      }

      StackFrame *prevFrame = (StackFrame *) &vm->stackItem( oldbase - VM_FRAME_SPACE );

      if( val >= 0 && val < prevFrame->m_param_count )
      {
         val = oldbase - prevFrame->m_param_count - VM_FRAME_SPACE + val;
         vm->regA().setBoolean( vm->stackItem( val ).isReference() ? true: false );
      }
      else {
         vm->regA().setBoolean( false );
      }
   }
}

/*#
   @function paramSet
   @inset varparams_support
   @brief Changes the nth paramter if it has been passed by reference.
   @param number the paramter to be changed (zero based)
   @param value the new value for the parameter
   @raise AccessError if @number is out of range.

   The function is equivalent to assigning the value directly to the required
   parameter; of course, in this way also optional parameter may be accessed.
   If the required parameter was passed by reference, also the original value
   in the caller is changed.

   If the given parameter number cannot be accessed, an AccessError is raised.
*/
FALCON_FUNC  paramSet ( ::Falcon::VMachine *vm )
{

   Item *number = vm->param(0);
   Item *value = vm->param(1);
   if ( number == 0 || ! number->isOrdinal() || value == 0) {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_range ).extra( "N,X" ) ) );
      return;
   }

   if ( vm->stackBase() == 0 )
   {
      vm->raiseRTError( new GenericError( ErrorParam( e_stackuf ) ) );
   }
   else
   {
      uint32 val = (uint32) number->forceInteger();

      StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
      uint32 oldbase = thisFrame->m_stack_base;
      if( oldbase == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_range ) ) );
         return;
      }

      StackFrame *prevFrame = (StackFrame *) &vm->stackItem( oldbase - VM_FRAME_SPACE );

      if( val >= 0 && val < prevFrame->m_param_count )
      {
         val = oldbase - prevFrame->m_param_count - VM_FRAME_SPACE + val;
         vm->stackItem( val ).dereference()->copy( *value );
      }
   }
}

}
}

/* end of param_ext.cpp */
