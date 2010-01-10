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

#include <string.h>

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
   StackFrame *thisFrame = vm->currentFrame();
   if( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );

   vm->retval( (int64) thisFrame->prev()->m_param_count );
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
      throw new ParamError( ErrorParam( e_param_range ).origin( e_orig_runtime ).extra( "(N)" ) );
   }

   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );

   // ...but we want the parameter count of our caller.
   StackFrame *prevFrame = thisFrame->prev();

   // ...while the parameters are below our frame's base.
   uint32 val = (uint32) number->forceInteger();
   if( val >= 0 && val < prevFrame->m_param_count )
   {
      vm->retval( prevFrame->m_params[val] );
   }
   else {
      vm->retnil();
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
   function will change locally the value of the parameter, but
   this value won't be reflected in the actual parameter that was used to
   call the function, unless the parameter was explicitly passed by reference.
   In some contexts, it may be useful to know if this is the case.

   If the given parameter number cannot be accessed, a AccessError is raised.
*/

FALCON_FUNC  paramIsRef ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( number == 0 || ! number->isOrdinal() ) {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ).extra( "N" ) );
   }

   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );

   // ...but we want the parameter count of our caller.
   StackFrame *prevFrame = thisFrame->prev();

   uint32 val = (uint32) number->forceInteger();
   if( val >= 0 && val < prevFrame->m_param_count )
   {
      vm->regA().setBoolean( prevFrame->m_params[val].isReference() ? true: false );
   }
   else {
      vm->regA().setBoolean( false );
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
      throw new ParamError( ErrorParam( e_param_range ).origin( e_orig_runtime ).extra( "N,X" ) );
   }

   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );

   // ...but we want the parameter count of our caller.
   StackFrame *prevFrame = thisFrame->prev();


   uint32 val = (uint32) number->forceInteger();
   if( val >= 0 && val < prevFrame->m_param_count )
   {
      prevFrame->m_params[val].dereference()->copy( *value );
   }
}


/*#
   @function argv
   @inset varparams_support
   @brief Returns all the parameters of the current function as a vector.

   If the current function doesn't receive any parameter, it returns nil.  
*/
FALCON_FUNC core_argv( VMachine *vm )
{
   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );

   // ...but we want the parameter count of our caller.
   StackFrame *prevFrame = thisFrame->prev();

   // ...while the parameters are below our frame's base.
   if( prevFrame->m_param_count > 0 )
   {
      CoreArray* arr = new CoreArray(prevFrame->m_param_count);
      Item* first = prevFrame->m_params;
      memcpy( arr->items().elements(), first, arr->items().esize( prevFrame->m_param_count ) );
      arr->length( prevFrame->m_param_count );
      vm->retval( arr );
   }
}

/*#
   @function argd
   @inset varparams_support
   @brief Returns a dictionary containing all the parameters passed to the current function.

   The dictionary contains the parameter names associated with the value passed by the caller.
   Parameters received beyond the officially declared ones aren't returned in this dictionary.
   
   If the function doesn't declare any parameter, returns nil.
*/
FALCON_FUNC core_argd( VMachine *vm )
{
   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );

   // ...but we want the parameter count of our caller.
   StackFrame *prevFrame = thisFrame->prev();
   
   // get the caller function symbol --- it holds the declared parameters
   const Symbol* sym = thisFrame->m_symbol;
   const Map* st =  sym->isFunction()? 
      &sym->getFuncDef()->symtab().map() :
      &sym->getExtFuncDef()->parameters()->map();
      
   CoreDict* ret = 0;
   Item* first = prevFrame->m_params;
      
   // ...while the parameters are below our frame's base.
   MapIterator iter = st->begin();
   while( iter.hasCurrent() )
   {
      Symbol *p = (*(Symbol**)iter.currentValue());
      if( p->isParam() )
      {
         if( ret == 0 )
            ret = new CoreDict( new LinearDict );
         ret->put( Item(new CoreString( p->name() )), first[p->itemId()] );
      }
      
     iter.next();
   }
   
   if ( ret != 0 )
      vm->retval( ret );
}

/*#
   @function passvp
   @inset varparams_support
   @brief Returns all the undeclared parameters, or passes them to a callable item
   @optparam citem Callable item on which to pass the parameters.
   @return An array containing unnamed parameters, or the return value \b citem.
   
   This function returns all the parameters passed to this function but not declared
   in its prototype (variable parameters) in an array.
   
   If the host function doesn't receive any extra parameter, this function returns
   an empty array. This is useful in case the array is immediately added to a direct
   call. For example:

   @code
   function receiver( a, b )
      > "A: ", a
      > "B: ", b
      > "Others: ", passvp().describe()
   end

   receiver( "one", "two", "three", "four" )
   @endcode
   
   If @b citem is specified, the function calls citem passing all the extra parameters
   to it. For example:

   @code
   function promptPrint( prompt )
      passvp( .[printl prompt] )
   end

   promptPrint( "The prompt: ", "arg1", " ", "arg2" )
   @endcode
*/
FALCON_FUNC core_passvp( VMachine *vm )
{
   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 || thisFrame->prev() == 0 )
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );
      
   // ok, do we have an item to call?
   Item* i_citem = vm->param(0);
   if ( i_citem != 0 && ! i_citem->isCallable() )
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra("[C]").origin( e_orig_runtime ) );
   
   // ...but we want the parameter count of our caller.
   StackFrame *prevFrame = thisFrame->prev();
   
   // get the caller function symbol --- it holds the declared parameters
   const Symbol* sym = thisFrame->m_symbol;
   unsigned size =  sym->isFunction()? 
      sym->getFuncDef()->symtab().size() :
      sym->getExtFuncDef()->parameters()->size();
   
   Item* first = prevFrame->m_params;
   int pcount = prevFrame->m_param_count - size;
   if ( pcount < 0 )
      pcount = 0;
   
   if ( i_citem )
   {
      while( size < prevFrame->m_param_count )
      {
         vm->pushParam( first[size] );
         ++size;
      }
      
      // size may be > param count in ext funcs.
      vm->callFrame(*i_citem, pcount );
   }
   else
   {
      CoreArray* arr = new CoreArray( pcount );
      while( size < prevFrame->m_param_count )
      {
         arr->append( first[size] );
         ++size;
      }

      vm->retval( arr );
   }
}


}
}

/* end of param_ext.cpp */
