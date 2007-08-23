/*
   FALCON - The Falcon Programming Language.
   FILE: core_func.cpp
   $Id: core_ext.cpp,v 1.50 2007/08/18 11:08:07 jonnymind Exp $

   Falcon module manager
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-01
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/


#include <falcon/module.h>
#include <falcon/runtime.h>
#include <falcon/item.h>
#include <falcon/types.h>
#include <falcon/stream.h>
#include <falcon/core_ext.h>
#include <falcon/error.h>
#include <falcon/vm.h>
#include <falcon/format.h>
#include "vmsema.h"

#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/cobject.h>
#include <falcon/cclass.h>
#include <falcon/pagedict.h>
#include <falcon/memory.h>
#include <falcon/error.h>
#include <falcon/sys.h>

#include <falcon/messages.h>
#include <falcon/engstrings.h>
#include <falcon/fbom.h>


namespace Falcon {

namespace core {

/****************************************
   VM Interface.
****************************************/

FALCON_FUNC  vmVersionInfo( ::Falcon::VMachine *vm )
{
   CoreArray *ca = new CoreArray( vm, 3 );
   ca->append( (int64) ((FALCON_ENGINE_VERSION_NUM >> 16) & 0xFF) );
   ca->append( (int64) ((FALCON_ENGINE_VERSION_NUM >> 8) & 0xFF) );
   ca->append( (int64) ((FALCON_ENGINE_VERSION_NUM ) & 0xFF) );
   vm->retval( ca );
}

FALCON_FUNC  vmVersionName( ::Falcon::VMachine *vm )
{
   String *str = new GarbageString( vm, FALCON_OFFICIAL_VERSION " (" FALCON_VERSION_NAME ")" );
   vm->retval( str );
}

/****************************************
   Generic item handling
****************************************/


/*@beginmodule core Direct support to language features and Virtual Machine interface */

/*@function len
   @param item an item of any kind
   @return an integer representing the lenght of the item

   @short Retreives the lenght of a collection

   The returned value represent the "size" of the item passed as a parameter.
   The number is consistent with the object type: in case of a string, it
   represents the count of characters, in case of arrays or dictionaries it
   represents the number of elements, in all the other cases the returned
   value is 1.

*/

FALCON_FUNC  len ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 ) {
      vm->retval( 0 );
      return;
   }

   Item *elem = vm->param(0);
   switch( elem->type() ) {
      case FLC_ITEM_STRING:
         vm->retval( (int) elem->asString()->length() );
      break;

      case FLC_ITEM_ARRAY:
         vm->retval( (int) elem->asArray()->length() );
      break;

      case FLC_ITEM_DICT:
         vm->retval( (int) elem->asDict()->length() );
      break;

      case FLC_ITEM_RANGE:
         vm->retval( 2 );
      break;

      default:
         vm->retval( 0 );
   }
}


/****************************************
   Error management
****************************************/

/** Error class constructor.
   Error( code, description, extra )
*/
FALCON_FUNC  Error_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();

   // subclasses may have already given a value to the userdata.
   Falcon::Error *err;
   if( einst->getUserData() == 0 )
   {
      err = new GenericError;
   }
   else {
      err = reinterpret_cast<ErrorCarrier *>(einst->getUserData())->error();
   }

   // declare that the script has created it
   err->origin( e_orig_script );
   vm->fillErrorContext( err );

   // filling properties
   Item *param = vm->param( 0 );
   if ( param != 0 && param->type() != FLC_ITEM_NIL  )
      err->errorCode( (int) param->forceInteger() );

   param = vm->param( 1 );
   if ( param != 0 && param->isString() )
      err->errorDescription( *param->asString() );

   param = vm->param( 2 );
   if ( param != 0 && param->isString() )
      err->extraDescription( *param->asString() );

   einst->setUserData( new ErrorCarrier( err ) );

   vm->retval( einst );
}

FALCON_FUNC  Error_toString ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   Falcon::ErrorCarrier *car = (Falcon::ErrorCarrier *) einst->getUserData();
   Falcon::Error *err = car->error();

   if ( err != 0 )
   {
      String *cs = new GarbageString( vm );
      err->toString( *cs );
      vm->retval( cs );
   }
   else
      vm->retnil();
}

FALCON_FUNC  Error_getSysErrDesc ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   Falcon::ErrorCarrier *car = (Falcon::ErrorCarrier *) einst->getUserData();
   Falcon::Error *err = car->error();

   if ( err != 0 )
   {
      String temp;
      ::Falcon::Sys::_describeError( err->systemError(), temp );
      vm->retval( temp );
   }
   else
      vm->retnil();
}



FALCON_FUNC  SyntaxError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::SyntaxError ) );

   Error_init( vm );
}


FALCON_FUNC  CodeError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::CodeError ) );

   Error_init( vm );
}

FALCON_FUNC  IoError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::IoError ) );

   Error_init( vm );
}

FALCON_FUNC  TypeError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::TypeError ) );

   Error_init( vm );
}


FALCON_FUNC  RangeError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::RangeError ) );

   Error_init( vm );
}

FALCON_FUNC  MathError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::MathError ) );

   Error_init( vm );
}

FALCON_FUNC  ParamError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::ParamError ) );

   Error_init( vm );
}

FALCON_FUNC  ParseError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::ParseError ) );

   Error_init( vm );
}

FALCON_FUNC  CloneError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new Falcon::CloneError ) );

   Error_init( vm );
}


/*@function int
   @param item The item to be converted

   @short Transforms the parameter in a integer.

   If the parameter is a string, a string-to-number coversion will be attempted.

*/
FALCON_FUNC  val_int ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 ) {
      vm->retnil();
      return;
   }

   Item *to_int = vm->param(0);

   switch( to_int->type() ) {
      case FLC_ITEM_INT:
      case FLC_ITEM_NUM:
         vm->retval( (int64)to_int->forceInteger() );
      break;

      case FLC_ITEM_STRING:
      {
         String *cs = to_int->asString();
         if ( cs->size() == 0 )
            vm->retval(0);
         else {
            int32 pos = cs->size() -1;
            if ( pos > 18 ) {
               vm->raiseRTError( new RangeError( ErrorParam( e_numparse_long ) ) );
               return;
            }
            uint32 chr =  cs->getCharAt( pos );
            uint64 val = 0;
            uint64 base = 1;
            while( pos > 0 ) {
               if ( chr < '0' || chr > '9' ) {
                  vm->raiseRTError( new RangeError( ErrorParam( e_numparse ) ) );
                  return;
               }
               val += ( chr -'0') * base;
               pos--;
               chr =  cs->getCharAt( pos );
               base *= 10;
            }
            if ( chr == '-' )
               vm->retval( -(int64)val );
            else {
               if ( chr < '0' || chr > '9' ) {
                  vm->raiseRTError( new RangeError( ErrorParam( e_numparse ) ) );
                  return;
               }

               vm->retval( (int64)(val + ( chr -'0' ) * base ) );
            }
         }
      }
      break;

      default:
         vm->raiseRTError( new RangeError( ErrorParam( e_numparse ) ) );
   }
}

/*@function typeOf
   @param item an item of any kind.
   @short Returns an integer indicating the type of an item.

   The value returned may be one of the following:<BR>
   <UL>
   <LI>NilType - the item is NIL</LI>
   <LI>IntegerType - the item is an integer</LI>
   <LI>NumericType - the item is a floating point number</LI>
   <LI>RangeType - the item is a range (a pair of two integers)</LI>
   <LI>FunctionType - the item is a function</LI>
   <LI>StringType - the item is a string </LI>
   <LI>ArrayType - the item is an array </LI>
   <LI>DictionaryType - the item is a dictionary</LI>
   <LI>ObjectType - the item is an object</LI>
   <LI>ClassType - the item is a class</LI>
   <LI>MethodType - the item is a method</LI>
   <LI>ExtMethodType - the item is a method provided by a native module</LI>
   <LI>ClassMethodType - the item is a method inside a class</LI>
   <LI>LibFuncType - the item is a function provided by a native module</LI>
   <LI>OpaqueType - the item is a user-defined opaque type</LI>
   <LI>LightType - the item is a user-defined light pointer</LI>
   </UL>
*/
FALCON_FUNC  typeOf ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
      vm->retnil();
   else
      vm->retval( vm->param( 0 )->type() );
}

/*@function isCallable
   @param item a possibly callable item.
   @return true if item could be called
*/
FALCON_FUNC  isCallable ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
      vm->retval( 0 );
   else
      vm->retval( vm->param( 0 )->isCallable() ? 1 : 0 );
}

/*@function getProperty
   @param item an object
   @param property a string naming a property
   @return the property
   @raise e_prop_acc if the property can't be found.
*/
FALCON_FUNC  getProperty( ::Falcon::VMachine *vm )
{
   Item *obj_x = vm->param(0);
   Item *prop_x = vm->param(1);

   if ( obj_x == 0 || ! obj_x->isObject() || prop_x == 0 || ! prop_x->isString() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( 0, S)" ) ) );
   }
   else if ( ! obj_x->asObject()->getProperty( *prop_x->asString(), vm->regA() ) )
   {
      vm->raiseRTError( new RangeError( ErrorParam( e_prop_acc ) ) );

   }
}

/*@function setProperty
   @param item an object
   @param property a string naming a property
   @param value an item that sets a new value
   @return the property
   @raise e_prop_acc if the property can't be found.
*/
FALCON_FUNC  setProperty( ::Falcon::VMachine *vm )
{
   Item *obj_x = vm->param(0);
   Item *prop_x = vm->param(1);
   Item *new_item = vm->param(2);

   if ( obj_x == 0 || ! obj_x->isObject() || prop_x == 0 || ! prop_x->isString() || new_item == 0) {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( getMessage( msg::core_002 ) ) ) );
   }
   else if ( ! obj_x->asObject()->setProperty( *prop_x->asString(), *new_item ) )
   {
      vm->raiseRTError( new RangeError( ErrorParam( e_prop_acc ) ) );
   }
}

/*@function exit
   @param value an item representing VM exit code.
   @short Requires immediate termination of the program.

   The program is immediately terminated and the toplevel VM loop is
   interrupted as soon as possible. In case of embedding applications,
   the exit value may be retreived by the embedder and interpreted as
   the "script return value"; in case of falcon command line compiler,
   the item is translated into an integer and provided as the exit
   value of the script.
*/

FALCON_FUNC  hexit ( ::Falcon::VMachine *vm )
{
   Item *ret = vm->param(0);

   vm->requestQuit();
   if ( ret != 0 )
      vm->retval( *ret );
}


/*@function chr
   @param code an UNICODE character ID.
   @return a single-char string.
   @short Converts a 0-255 integer in the corresponding character.

   @see ord
*/

FALCON_FUNC  chr ( ::Falcon::VMachine *vm )
{
   uint32 val;
   Item *elem = vm->param(0);
   if ( elem == 0 ) return;
   if ( elem->type() == FLC_ITEM_INT )
      val = (uint32) elem->asInteger();
   else if ( elem->type() == FLC_ITEM_NUM )
      val = (uint32) elem->asNumeric();
   else {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "(N)" ) ) );
      return;
   }

   String *ret = new GarbageString( vm );
   ret->append( val );
   vm->retval( ret );
}

/*@function ord
   @param string a string
   @return the UNICODE value of the first element in the string.
   @short Returns the ASCII value of the first element in the string.

   @todo add international support. (?) move this out of core.
   @see chr
*/
FALCON_FUNC  ord ( ::Falcon::VMachine *vm )
{
   Item *elem = vm->param(0);
   if ( elem == 0 || ! elem->isString() || elem->asString()->size() == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params).extra( "(N)" ) ) );
      return;
   }

   vm->retval( (int64) elem->asString()->getCharAt(0) );
}

/*@function toString
   @param item an item to be converted to string.
   @optparam deccount number of significative decimals for numeric items.
   @return the string representation of the item.

   @short Returns a string representation of the item.

   If the item is a number, the second parameter will determine how many
   decimals will be printed. If it is an object, and if it provides a
   toString method, that method will be called.

*/

FALCON_FUNC  hToString ( ::Falcon::VMachine *vm )
{
   Item *elem = vm->param(0);
   Item *format = vm->param(1);

   Fbom::toString( vm, elem, format );
}

/*@begingroup varparm Variable Parameter management.
   Falcon supports variable parameter calling; a function or method may access
   the items that have been used in the parameter call by counting them and
   accessing them one by one.

   Parameter passed by reference may be modified with the appropriate function.

   This functions may be used wether the calling function provides a list of formal
   paramters or not. The first formal parameter will be treated as the variable
   parameter number zero, and the paramter count may be the same as, more than or less than
   the number of formal parameters. So, part of the paramters may be accessible via
   paramter names, and the others may be accessed with this functions.

*/

/*@function paramCount
   @return the parameter count
   @short Returns number of parameter that have been passed to the current function or method.
*/

FALCON_FUNC  paramCount ( ::Falcon::VMachine *vm )
{
   // temporarily save the call environment.
   if ( vm->stackBase() == 0 ) {
      vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
   }
   else {
      StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
      if( thisFrame->m_stack_base == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
         return;
      }

      StackFrame *prevFrame = (StackFrame *) &vm->stackItem( thisFrame->m_stack_base - VM_FRAME_SPACE );
      vm->retval( prevFrame->m_param_count );
   }
}

/*@function paramNumber
   @short get the Nth parameter
   @param the paremeter that must be returned, zero based
   @return the nth paramter (zero based) or NIL if the parameter is not given
*/

FALCON_FUNC  paramNumber ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( number == 0 || ! number->isOrdinal() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_outside ).extra( "(N)" ) ) );
      return;
   }

   if ( vm->stackBase() == 0 )
   {
      vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
   }
   else {
      int32 val = (int32) number->forceInteger();

      StackFrame *thisFrame = (StackFrame *) vm->currentStack().at( vm->stackBase() - VM_FRAME_SPACE );
      uint32 oldbase = thisFrame->m_stack_base;
      if( oldbase == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
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

/*@function paramIsRef
   @short check whether the nth parameter has been passed by value or by reference
   @param number the paramter that must be checked (zero based)
   @return true if the parameter has been passed by reference, false otherwise
*/

FALCON_FUNC  paramIsRef ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( number == 0 || ! number->isOrdinal() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_outside ).extra( "(N)" ) ) );
      return;
   }

   if ( vm->stackBase() == 0 )
   {
      vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
   }
   else
   {
      int32 val = (int32) number->forceInteger();

      StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
      uint32 oldbase = thisFrame->m_stack_base;
      if( oldbase == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
         return;
      }

      StackFrame *prevFrame = (StackFrame *) &vm->stackItem( oldbase - VM_FRAME_SPACE );

      if( val >= 0 && val < prevFrame->m_param_count )
      {
         val = oldbase - prevFrame->m_param_count - VM_FRAME_SPACE + val;
         vm->retval( vm->stackItem( val ).isReference() ? (int64) 1 : (int64) 0 );
      }
      else {
         vm->retval( (int64) 0 );
      }
   }
}

/*@function paramSet
   @short Changes the nth paramter if it has been passed by reference.
   @param number the paramter to be changed (zero based)
   @param value the new value for the parameter

   In case of explicit parameter list, it is possible to change a paramter that
   has been passed by reference by just assigning a new value to it; but when
   the list is not explicit, that is, when variable paramters are provided to the
   called item, this function allows to provide the caller with a changed paramter
   value.
*/
FALCON_FUNC  paramSet ( ::Falcon::VMachine *vm )
{

   Item *number = vm->param(0);
   Item *value = vm->param(1);
   if ( number == 0 || ! number->isOrdinal() || value == 0) {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_outside ).extra( "( N, ? )" ) ) );
      return;
   }

   if ( vm->stackBase() == 0 )
   {
      vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
   }
   else
   {
      int32 val = (int32) number->forceInteger();

      StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
      uint32 oldbase = thisFrame->m_stack_base;
      if( oldbase == 0 ) {
         vm->raiseRTError( new GenericError( ErrorParam( e_param_outside ) ) );
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

/*@endgroup */

/*@begingroup coro_sup Coroutine support
   The functions in this group allows to interact with the coroutine support that is
   provided by the Virtual Machine. Most of them translate in requests to the virtual
   machine.
*/

/*@function yield
   @short gives up the rest of the coroutine time slice.

   The calling coroutine is immediately swapped out and put at the end of the
   ready coroutines waiting to be served. In case there aren't any other
   coroutines ready to be executed, the function does nothing.

*/
FALCON_FUNC  yield ( ::Falcon::VMachine *vm )
{
   vm->yieldRequest( 0.0 );
}

/*@function yieldOut
   @short Requires termination of the current coroutine.
   @param retval a return value for the coroutine.

   The calling coroutine is immediately terminated

   @see exit
*/
FALCON_FUNC  yieldOut ( ::Falcon::VMachine *vm )
{
   Item *ret = vm->param(0);
   vm->yieldRequest( -1.0 );
   if ( ret != 0 )
      vm->retval( *ret );
   else
      vm->retnil();
}


FALCON_FUNC  _f_sleep ( ::Falcon::VMachine *vm )
{
   Item *amount = vm->param(0);
   numeric pause;
   if( amount == 0 )
      pause = 0.0;
   else {
      pause = amount->forceNumeric();
      if ( pause < 0.0 )
         pause = 0.0;
   }

   vm->yieldRequest( pause );
}

FALCON_FUNC  beginCritical ( ::Falcon::VMachine *vm )
{
   vm->allowYield( false );
}

FALCON_FUNC  endCritical ( ::Falcon::VMachine *vm )
{
   vm->allowYield( true );
}

FALCON_FUNC  Semaphore_init ( ::Falcon::VMachine *vm )
{
   Item *qty = vm->param(0);
   int32 value = 0;
   if ( qty != 0 ) {
      if ( qty->type() == FLC_ITEM_INT )
         value = (int32) qty->asInteger();
      else if ( qty->type() == FLC_ITEM_NUM )
         value = (int32) qty->asNumeric();
      else {
         vm->raiseRTError( new ParamError( ErrorParam( e_param_outside ).extra( "( N )" ) ) );
         return;
      }
   }

   VMSemaphore *sem = new VMSemaphore( value );
   vm->self().asObject()->setUserData( sem );
}

FALCON_FUNC  Semaphore_post ( ::Falcon::VMachine *vm )
{
   VMSemaphore *semaphore = static_cast< VMSemaphore *>(vm->self().asObject()->getUserData());
   Item *qty = vm->param(0);
   int32 value = 1;
   if ( qty != 0 ) {
      if ( qty->type() == FLC_ITEM_INT )
         value = (int32)qty->asInteger();
      else if ( qty->type() == FLC_ITEM_NUM )
         value = (int32) qty->asNumeric();
      else {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( N )" ) ) );
         return;
      }
      if (value <= 0)
         value = 1;
   }

   semaphore->post( vm, value );
}

FALCON_FUNC  Semaphore_wait ( ::Falcon::VMachine *vm )
{
   VMSemaphore *semaphore = static_cast< VMSemaphore *>(vm->self().asObject()->getUserData());
   semaphore->wait( vm );
}

FALCON_FUNC vmSuspend( ::Falcon::VMachine *vm )
{
   vm->requestSuspend();
}

/****************************************
   The Format class.
****************************************/

FALCON_FUNC  Format_parse ( ::Falcon::VMachine *vm )
{

   CoreObject *einst = vm->self().asObject();
   Format *fmt = (Format *) einst->getUserData();

   Item *param = vm->param( 0 );
   if ( param != 0 )
   {
      if( ! param->isString() )
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "[S]" ) ) );
      }
      else  {
         fmt->parse( *param->asString() );
         if( ! fmt->isValid() )
         {
            vm->raiseRTError( new ParseError( ErrorParam( e_param_fmt_code ) ) );
         }
      }
   }
}

FALCON_FUNC  Format_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();

   Format *fmt = new Format;
   einst->setUserData( fmt );

   Format_parse( vm );
}


FALCON_FUNC  Format_format ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   Format *fmt = (Format *) einst->getUserData();

   Item *param = vm->param( 0 );
   Item *dest = vm->param( 1 );
   if( param == 0 || ( dest != 0 && ! dest->isString() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X,[S]" ) ) );
   }
   else
   {
      String *tgt;

      if( dest != 0 )
      {
         tgt = dest->asString();
      }
      else {
         tgt = new GarbageString( vm );
      }

      if( ! fmt->format( vm, *param, *tgt ) )
         vm->retnil();
      else
         vm->retval( tgt );
   }
}

FALCON_FUNC  Format_toString ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   Format *fmt = (Format *) einst->getUserData();
   vm->retval( new GarbageString( vm,fmt->originalFormat()) );
}

//========================================================
// The command line parser
//========================================================

FALCON_FUNC  CmdlineParser_parse( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_params = vm->param( 0 );

   if ( i_params == 0 )
   {
      // get the parameters from the VM args object
      i_params = vm->findGlobalItem( "args" );
      if ( i_params == 0 || ! i_params->isArray() ) {
         vm->raiseRTError( new CodeError( ErrorParam( e_undef_sym ).extra( "args" ).hard() ) );
         return;
      }
   }
   else if ( ! i_params->isArray() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( A )" ) ) );
      return;
   }

   CoreArray *args = i_params->asArray();

   // zero request.
   self->setProperty( "_request", (int64) 0 );
   self->setProperty( "lastParsed", (int64) 0 );

   // status.
   typedef enum {
      t_none,
      t_waitingValue,
      t_allFree
   } t_states;

   t_states state = t_none ;
   String currentOption;
   Item i_method;
   Item i_passMM;
   self->getProperty( "passMinusMinus", i_passMM );
   bool passMM = i_passMM.isTrue();
   Item _request;
   String subParam;
   uint32 i;

   for ( i = 0; i < args->length(); i++ )
   {
      Item &i_opt = args->at( i );
      if ( !i_opt.isString() )
      {
         vm->raiseRTError(
            new ParamError( ErrorParam( e_param_type ).
                  extra( getMessage( msg::core_002 ) )
               )
            );

         return;
      }

      String &opt = *i_opt.asString();
       // if we were expecting a value, we MUST consider ANYTHING as it was a value.
      if ( state == t_waitingValue )
      {
         self->getProperty( "onValue", i_method );
         if ( i_method.methodize( self ) )
         {
            vm->pushParameter( &currentOption );
            vm->pushParameter( i_opt );
            vm->callItem( i_method, 2 );
            if( vm->hadError() )
               return;

            vm->resetEvent();
            state = t_none;
         }
         else
         {
            vm->retval( false );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
      else if( opt.length() == 0 || (opt.getCharAt( 0 ) != '-' || opt.length() == 1) || state == t_allFree )
      {

         self->getProperty( "onFree", i_method );
         if ( i_method.methodize( self ) )
         {
            vm->pushParameter( i_opt );
            vm->callItem( i_method, 1 );
            if( vm->hadError() )
               return;
            vm->resetEvent();

         }
         else
         {
            vm->retval( false );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
      else if ( opt == "--" && ! passMM )
      {
         state = t_allFree;
         continue; // to skip return value.
      }
      else {
         // we have at least one '-', and length > 1
         if ( opt.getCharAt( 1 ) == (uint32) '-' )
         {
            self->getProperty( "onOption", i_method );

            if ( i_method.methodize( self ) )
            {
               if ( passMM && opt.size() == 2 )
                  vm->pushParameter( i_opt );
               else {
                  //Minimal optimization; reuse the same string and memory
                  subParam = opt.subString( 2 );
                  vm->pushParameter( &subParam );
               }

               vm->callItem( i_method, 1 );
               if( vm->hadError() )
                  return;
               vm->resetEvent();
               self->getProperty( "_request", _request );
               // value requested?
               if ( _request.asInteger() == 1 ) {
                  currentOption = subParam;
               }
            }
            else
            {
               vm->retval( false );
               self->setProperty( "lastParsed", (int64) i );
               return;
            }
         }
         else {
            // we have a switch set.
            for( uint32 chNum = 1; chNum < opt.length(); chNum++ )
            {
               //Minimal optimization; reuse the same string and memory

               subParam.size( 0 );
               subParam.append( opt.getCharAt( chNum ) );

               if ( chNum < opt.length() -1 && opt.getCharAt( chNum +1 ) == (uint32) '-' )
               {
                  // switch turnoff.
                  self->getProperty( "onSwitchOff", i_method );
                  if ( i_method.methodize( self ) )
                  {
                     vm->pushParameter( &subParam );
                     vm->callItem( i_method, 1 );
                     if( vm->hadError() )
                        return;
                     vm->resetEvent();
                 }
                  else
                  {
                     vm->retval( false );
                     self->setProperty( "lastParsed", (int64) i );
                     return;
                  }
                  chNum++;
               }
               else {
                  self->getProperty( "onOption", i_method );
                  if ( i_method.methodize( self ) )
                  {
                     vm->pushParameter( &subParam );
                     vm->callItem( i_method, 1 );
                     if( vm->hadError() )
                        return;
                     vm->resetEvent();
                  }
                  else
                  {
                     vm->retval( false );
                     self->setProperty( "lastParsed", (int64) i );
                     return;
                  }
               }

               self->getProperty( "_request", _request );
               // value requested?
               if ( _request.asInteger() == 1 ) {
                  currentOption = subParam;
               }
            }
         }

         self->getProperty( "_request", _request );
         // value requested?
         if ( _request.asInteger() == 1 ) {
            state = t_waitingValue;
            self->setProperty( "_request", (int64) 0 );
         }
         // or request to terminate?
         else if ( _request.asInteger() == 2 )
         {
            self->setProperty( "_request", (int64) 0 );
            vm->retval( true );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
   }

   self->setProperty( "lastParsed", (int64) i );
   vm->resetEvent();
   vm->retval( true );
}

FALCON_FUNC  CmdlineParser_expectValue( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "_request", (int64) 1 );
}

FALCON_FUNC  CmdlineParser_terminate( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "_request", (int64) 2 );
}

FALCON_FUNC  CmdlineParser_usage( ::Falcon::VMachine *vm )
{
   vm->stdErr()->writeString( "The stub for \"CmdlineParser.usage()\" has been called.\n" );
   vm->stdErr()->writeString( "This class should be derived and the method usage() overloaded.\n" );
}

// Garbage Collector control

FALCON_FUNC  gcEnable( ::Falcon::VMachine *vm )
{
   if( vm->param(0) == 0 )
      vm->retval( vm->memPool()->autoCleanMode() ? 1 : 0 );
   else
      vm->memPool()->autoCleanMode( vm->param(0)->isTrue() );
}

FALCON_FUNC  gcSetThreshold( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param( 0 );
   Item *p1 = vm->param( 1 );
   bool done = false;

   if( p0 != 0 && p0->isOrdinal() ) {
      done = true;
      vm->memPool()->thresholdMemory( (uint32) p0->forceInteger() );
   }

   if( p1 != 0 && p1->isOrdinal() ) {
      done = true;
      vm->memPool()->reclaimLevel( (uint32) p1->forceInteger() );
   }

   if ( ! done )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( [N], [N] )" ) ) );
   }
}

FALCON_FUNC  gcSetTimeout( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param( 0 );
   bool done = false;

   if( p0 != 0 && p0->isOrdinal() ) {
      vm->memPool()->setTimeout( (uint32) p0->forceInteger() );
   }
   else
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( N )" ) ) );
   }
}

FALCON_FUNC  gcPerform( ::Falcon::VMachine *vm )
{
   bool bRec;

   if ( vm->param( 0 ) != 0 )
   {
      bRec = vm->param( 0 )->isTrue();
   }
   else {
      bRec = false;
   }

   vm->retval( vm->memPool()->performGC( bRec ) ? 1 : 0 );
}

FALCON_FUNC  gcGetParams( ::Falcon::VMachine *vm )
{
   Item *i_mpAllocMem = vm->param( 0 );
   Item *i_mpAllocItems = vm->param( 1 );
   Item *i_mpAliveMem = vm->param( 2 );
   Item *i_mpAliveItems = vm->param( 3 );
   Item *i_mpThreshold = vm->param( 4 );
   Item *i_mpRecLev = vm->param( 5 );
   Item *i_mpTimeout = vm->param( 6 );

   if( i_mpAllocMem != 0 )
      i_mpAllocMem->setInteger( vm->memPool()->allocatedMem() );
   if( i_mpAllocItems != 0 )
      i_mpAllocItems->setInteger( vm->memPool()->allocatedItems() );
   if( i_mpAliveMem != 0 )
      i_mpAliveMem->setInteger( vm->memPool()->aliveMem() );
   if( i_mpAliveItems != 0 )
      i_mpAliveItems->setInteger( vm->memPool()->aliveItems() );
   if( i_mpThreshold != 0 )
      i_mpThreshold->setInteger( vm->memPool()->thresholdMemory() );
   if( i_mpRecLev != 0 )
      i_mpRecLev->setInteger( vm->memPool()->reclaimLevel() );
   if( i_mpTimeout != 0 )
      i_mpTimeout->setInteger( vm->memPool()->getTimeout() );
}

/****************************************
   The iterator class
****************************************/

FALCON_FUNC  Iterator_init( ::Falcon::VMachine *vm )
{
   Item *collection = vm->param(0);
   Item *pos = vm->param(1);

   if( collection == 0 || ( pos != 0 && ! pos->isOrdinal() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X,[N]" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   int32 p = pos == 0 ? 0: (int32) pos->forceInteger();

   switch( collection->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item tgt;
         String *orig = collection->asString();
         vm->referenceItem( tgt, *collection );
         self->setProperty( "origin", tgt );

         if( orig->checkPosBound( p ) )
         {
            self->setProperty( "_pos", (int64) p );
         }
         else {
            vm->raiseRTError( new RangeError( ErrorParam( e_inv_params ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *orig = collection->asArray();
         self->setProperty( "origin", *collection );
         if( orig->checkPosBound( p ) )
         {
            self->setProperty( "_pos", (int64) p );
         }
         else {
            vm->raiseRTError( new RangeError( ErrorParam( e_inv_params ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *orig = collection->asDict();
         self->setProperty( "origin", *collection );
         DictIterator *iter;
         if( p == 0 )
            iter = orig->begin();
         else if( p == -1 )
            iter = orig->last();
         else {
            vm->raiseRTError( new RangeError( ErrorParam( e_inv_params ) ) );
            return;
         }

         self->setUserData( iter );
      }
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ) ) );
         return;
   }
}

FALCON_FUNC  Iterator_hasCurrent( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         vm->retval( (int64) (p < porigin->asString()->length() ? 1: 0 ) );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         vm->retval( (int64) (p < porigin->asArray()->length() ? 1: 0 ) );
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         vm->retval( (int64) ( iter->isValid() ? 1: 0 ) );
      }
      break;

      default:
         // raise an exception???
         vm->retval( (int64) 0 );
   }
}

FALCON_FUNC  Iterator_hasNext( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         vm->retval( (int64) (p + 1 < porigin->asString()->length() ? 1: 0 ) );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         vm->retval( (int64) (p + 1 < porigin->asArray()->length() ? 1: 0 ) );
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         vm->retval( (int64) ( iter->hasNext() ? 1: 0 ) );
      }
      break;

      default:
         // raise an exception???
         vm->retval( (int64) 0 );
   }
}

FALCON_FUNC  Iterator_hasPrev( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         vm->retval( (int64) (p > 0 && p != String::npos ? 1: 0 ) );
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         vm->retval( (int64) ( iter->hasPrev() ? 1: 0 ) );
      }
      break;

      default:
         // raise an exception???
         vm->retval( (int64) 0 );
   }
}

FALCON_FUNC  Iterator_next( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger() + 1;
         vm->retval( (int64) (p < porigin->asString()->length() ? 1: 0 ) );
         self->setProperty( "_pos", (int64) p );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger() + 1;
         vm->retval( (int64) (p < porigin->asArray()->length() ? 1: 0 ) );
         self->setProperty( "_pos", (int64) p );
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         vm->retval( (int64) ( iter->next() ? 1: 0 ) );
      }
      break;

      default:
         // raise an exception???
         vm->retval( (int64) 0 );
   }
}

FALCON_FUNC  Iterator_prev( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         if( p > 0 )
         {
            vm->retval( (int64) 1 );
            self->setProperty( "_pos", (int64) p - 1 );
         }
         else  {
            p = String::npos; // ok also for arrays
            self->setProperty( "_pos", (int64) p );
            vm->retval( (int64) 0 );
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         vm->retval( (int64) ( iter->prev() ? 1: 0 ) );
      }
      break;

      default:
         // raise an exception???
         vm->retval( (int64) 0 );
   }
}

FALCON_FUNC  Iterator_value( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();
   Item *subst = vm->param( 0 );

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         if( p < porigin->asString()->length() )
         {
            GarbageString *str = new GarbageString( vm,
               porigin->asString()->subString( p, p + 1 ) );
            vm->retval( str );

            // change value
            if( subst != 0 )
            {
               switch( subst->type() )
               {
                  case FLC_ITEM_STRING:
                     porigin->asString()->change( p, p + 1, subst->asString() );
                  break;

                  case FLC_ITEM_NUM:
                     porigin->asString()->setCharAt( p, (uint32) subst->asNumeric() );
                  break;

                  case FLC_ITEM_INT:
                     porigin->asString()->setCharAt( p, (uint32) subst->asInteger() );
                  break;

                  default:
                     vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "S/N" ) ) );
               }
            }
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         if( p < porigin->asArray()->length() )
         {
            vm->retval( porigin->asArray()->at( p ) );
            // change value
            if( subst != 0 )
            {
               porigin->asArray()->at( p ) = *subst;
            }
            return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         if( iter->isValid() )
         {
            vm->retval( iter->getCurrent() );
            // change value
            if( subst != 0 )
            {
               iter->getCurrent() = *subst;
            }

            return;
         }
      }
      break;
   }

   vm->raiseRTError( new RangeError( ErrorParam( e_arracc ) ) );
}

FALCON_FUNC  Iterator_key( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   if( origin.isDict() )
   {
      DictIterator *iter = (DictIterator *) self->getUserData();
      if( iter->isValid() )
      {
         vm->retval( iter->getCurrentKey() );
         return;
      }
   }

   vm->raiseRTError( new RangeError( ErrorParam( e_arracc ).extra( "missing key" ) ) );
}

FALCON_FUNC  Iterator_equal( ::Falcon::VMachine *vm )
{
   Item *i_other = vm->param(0);

   if( i_other == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "O" ) ) );
      return;
   }

   if( i_other->isObject() )
   {
      CoreObject *other = i_other->asObject();

      if( other->derivedFrom( "Iterator" ) )
      {
         CoreObject *self = vm->self().asObject();

         Item origin, other_origin;
         self->getProperty( "origin", origin );
         other->getProperty( "origin", other_origin );
         if( origin.dereference()->equal( *other_origin.dereference() ) )
         {
            switch( origin.type() )
            {
               case FLC_ITEM_STRING:
               case FLC_ITEM_REFERENCE:
               case FLC_ITEM_ARRAY:
               {
                  Item pos1, pos2;
                  self->getProperty( "_pos", pos1 );
                  other->getProperty( "_pos", pos2 );
                  if( pos1 == pos2 )
                  {
                     vm->retval( (int64) 1 );
                     return;
                  }
               }
               break;

               case FLC_ITEM_DICT:
               {
                  DictIterator *iter = (DictIterator *) self->getUserData();
                  DictIterator *other_iter = (DictIterator *) other->getUserData();
                  if( iter->equal( *other_iter ) )
                  {
                     vm->retval( (int64) 1 );
                     return;
                  }
               }
               break;

            }
         }
      }
   }

   vm->retval( (int64) 0 );
}


FALCON_FUNC  Iterator_clone( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   CoreIterator *iter = (CoreIterator *) self->getUserData();


   // create an instance
   Item *i_cls = vm->findGlobalItem( "Iterator" );
   fassert( i_cls != 0 );
   CoreObject *other = i_cls->asClass()->createInstance();

   // copy low level iterator, if we have one
   if ( iter != 0 ) {
      other->setUserData( iter->clone() );
   }

   // then copy properties
   Item prop;
   self->getProperty( "origin", prop );
   other->setProperty( "origin", prop );
   self->getProperty( "_pos", prop );
   other->setProperty( "_pos", prop );

   // we can return the object
   vm->retval( other );
}

FALCON_FUNC  Iterator_remove( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         String *str = porigin->asString();

         if ( p < str->length() )
         {
            str->remove( p, 1 );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         CoreArray *array = porigin->asArray();

         if ( p < array->length() )
         {
            array->remove( p );
            return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         DictIterator *iter = (DictIterator *) self->getUserData();
         CoreDict *dict = porigin->asDict();

         if( iter->isValid() )
         {
            dict->remove( iter );
            return;
         }
      }
      break;
   }

   vm->raiseRTError( new RangeError( ErrorParam( e_arracc ) ) );
}

FALCON_FUNC  PageDict( ::Falcon::VMachine *vm )
{
   Item *i_pageSize = vm->param(0);

   if( i_pageSize != 0 && ! i_pageSize->isOrdinal() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N" ) ) );
      return;
   }

   uint32 pageSize = (uint32)( i_pageSize == 0 ? 33 : (uint32)i_pageSize->forceInteger() );
   CoreDict *cd = new ::Falcon::PageDict( vm, pageSize );
   vm->retval( cd );
}

} // end of core namespace


/****************************************
   Module initializer
****************************************/

Module * core_module_init()
{
   Module *core = new Module();
   core->name( "falcon.core" );

   core->addGlobal( "args", true );
   core->addGlobal( "scriptName", true );
   core->addGlobal( "scriptPath", true );

   core->addExtFunc( "len", Falcon::core::len );
   core->addExtFunc( "chr", Falcon::core::chr );
   core->addExtFunc( "ord", Falcon::core::ord );
   core->addExtFunc( "toString", Falcon::core::hToString );
   core->addExtFunc( "isCallable", Falcon::core::isCallable );
   core->addExtFunc( "getProperty", Falcon::core::getProperty );
   core->addExtFunc( "setProperty", Falcon::core::setProperty );

   core->addExtFunc( "yield", Falcon::core::yield );
   core->addExtFunc( "yieldOut", Falcon::core::yieldOut );
   core->addExtFunc( "sleep", Falcon::core::_f_sleep );
   core->addExtFunc( "beginCritical", Falcon::core::beginCritical );
   core->addExtFunc( "endCritical", Falcon::core::endCritical );
   core->addExtFunc( "suspend", Falcon::core::vmSuspend );

   core->addExtFunc( "int", Falcon::core::val_int );
   core->addExtFunc( "typeOf", Falcon::core::typeOf );
   core->addExtFunc( "exit", Falcon::core::hexit );

   core->addExtFunc( "paramCount", Falcon::core::paramCount );
   core->addExtFunc( "paramNumber", Falcon::core::paramNumber );
   core->addExtFunc( "paramIsRef", Falcon::core::paramIsRef );
   core->addExtFunc( "paramSet", Falcon::core::paramSet );
   core->addExtFunc( "PageDict", Falcon::core::PageDict );

   // Creating the TraceStep class:
   // ... first the constructor
   /*Symbol *ts_init = core->addExtFunc( "TraceStep._init", Falcon::core::TraceStep_init );

   //... then the class
   Symbol *ts_class = core->addClass( "TraceStep", ts_init );

   // then add var props; flc_CLSYM_VAR is 0 and is linked correctly by the VM.
   core->addClassProperty( ts_class, "module" );
   core->addClassProperty( ts_class, "symbol" );
   core->addClassProperty( ts_class, "pc" );
   core->addClassProperty( ts_class, "line" );
   // ... finally add a method, using the symbol that this module returns.
   core->addClassMethod( ts_class, "toString",
      core->addExtFunc( "TraceStep.toString", Falcon::core::TraceStep_toString ) );*/

   // Creating the Error class class:
   Symbol *error_init = core->addExtFunc( "Error._init", Falcon::core::Error_init );
   Symbol *error_class = core->addClass( "Error", error_init );
   core->addClassMethod( error_class, "toString",
         core->addExtFunc( "Error.toString", Falcon::core::Error_toString ) );
   core->addClassProperty( error_class, "code" );
   core->addClassProperty( error_class, "description" );
   core->addClassProperty( error_class, "message" );
   core->addClassProperty( error_class, "systemError" );
   core->addClassProperty( error_class, "origin" );
   core->addClassProperty( error_class, "module" );
   core->addClassProperty( error_class, "symbol" );
   core->addClassProperty( error_class, "line" );
   core->addClassProperty( error_class, "pc" );
   core->addClassProperty( error_class, "subErrors" );
   core->addClassMethod( error_class, "getSysErrorDesc", Falcon::core::Error_getSysErrDesc );

   // Other derived error classes.
   Falcon::Symbol *synerr_cls = core->addClass( "SyntaxError", Falcon::core::SyntaxError_init );
   synerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *codeerr_cls = core->addClass( "CodeError", Falcon::core::CodeError_init );
   codeerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *rangeerr_cls = core->addClass( "RangeError", Falcon::core::RangeError_init );
   rangeerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *matherr_cls = core->addClass( "MathError", Falcon::core::MathError_init );
   matherr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *ioerr_cls = core->addClass( "IoError", Falcon::core::IoError_init );
   ioerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *typeerr_cls = core->addClass( "TypeError", Falcon::core::TypeError_init );
   typeerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *paramerr_cls = core->addClass( "ParamError", Falcon::core::ParamError_init );
   paramerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *parsererr_cls = core->addClass( "ParseError", Falcon::core::ParseError_init );
   parsererr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   Falcon::Symbol *cloneerr_cls = core->addClass( "CloneError", Falcon::core::CloneError_init );
   cloneerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   //=========================================

   // Creating the semaphore class
   Symbol *semaphore_init = core->addExtFunc( "Semaphore._init", Falcon::core::Semaphore_init );
   Symbol *semaphore_class = core->addClass( "Semaphore", semaphore_init );

   core->addClassMethod( semaphore_class, "post",
            core->addExtFunc( "Semaphore.post", Falcon::core::Semaphore_post ) );
   core->addClassMethod( semaphore_class, "wait",
            core->addExtFunc( "Semaphore.wait", Falcon::core::Semaphore_wait ) );

   // The command line parser class
   Symbol *cmdparser_class = core->addClass( "CmdlineParser", true );
   core->addClassMethod( cmdparser_class, "parse", Falcon::core::CmdlineParser_parse );
   core->addClassMethod( cmdparser_class, "expectValue", Falcon::core::CmdlineParser_expectValue );
   core->addClassMethod( cmdparser_class, "terminate", Falcon::core::CmdlineParser_terminate );
   // private property internally used to communicate between the child classes and
   // the base parse.
   core->addClassProperty( cmdparser_class, "_request" );
   // Properties that will hold callbacks
   core->addClassProperty( cmdparser_class, "onOption" );
   core->addClassProperty( cmdparser_class, "onFree" );
   core->addClassProperty( cmdparser_class, "onValue" );
   core->addClassProperty( cmdparser_class, "onSwitchOff" );
   core->addClassProperty( cmdparser_class, "passMinusMinus" );
   core->addClassProperty( cmdparser_class, "lastParsed" );
   core->addClassMethod( cmdparser_class, "usage", Falcon::core::CmdlineParser_usage );

   // GC support
   core->addExtFunc( "gcEnable", Falcon::core::gcEnable );
   core->addExtFunc( "gcSetThreshold", Falcon::core::gcSetThreshold );
   core->addExtFunc( "gcPerform", Falcon::core::gcPerform );
   core->addExtFunc( "gcSetTimeout", Falcon::core::gcSetTimeout );
   core->addExtFunc( "gcGetParams", Falcon::core::gcGetParams );

   // VM support
   core->addExtFunc( "vmVersionInfo", Falcon::core::vmVersionInfo );
   core->addExtFunc( "vmVersionName", Falcon::core::vmVersionName );

   // Format
   Symbol *format_class = core->addClass( "Format", Falcon::core::Format_init );
   core->addClassMethod( format_class, "format", Falcon::core::Format_format );
   core->addClassMethod( format_class, "parse", Falcon::core::Format_parse );
   core->addClassMethod( format_class, "toString", Falcon::core::Format_toString );
   core->addClassProperty( format_class,"size" );
   core->addClassProperty( format_class, "decimals" );
   core->addClassProperty( format_class, "paddingChr" );
   core->addClassProperty( format_class, "groupingChr" );
   core->addClassProperty( format_class, "decimalChr" );
   core->addClassProperty( format_class, "grouiping" );
   core->addClassProperty( format_class, "fixedSize" );
   core->addClassProperty( format_class, "rightAlign" );
   core->addClassProperty( format_class, "originalFormat" );
   core->addClassProperty( format_class, "misAct" );
   core->addClassProperty( format_class, "convType" );
   core->addClassProperty( format_class, "nilFormat" );
   core->addClassProperty( format_class, "negFormat" );
   core->addClassProperty( format_class, "numFormat" );

   // Iterators
   Symbol *iterator_class = core->addClass( "Iterator", Falcon::core::Iterator_init );
   core->addClassMethod( iterator_class, "hasCurrent", Falcon::core::Iterator_hasCurrent );
   core->addClassMethod( iterator_class, "hasNext", Falcon::core::Iterator_hasNext );
   core->addClassMethod( iterator_class, "hasPrev", Falcon::core::Iterator_hasPrev );
   core->addClassMethod( iterator_class, "next", Falcon::core::Iterator_next );
   core->addClassMethod( iterator_class, "prev", Falcon::core::Iterator_prev );
   core->addClassMethod( iterator_class, "value", Falcon::core::Iterator_value );
   core->addClassMethod( iterator_class, "key", Falcon::core::Iterator_key );
   core->addClassMethod( iterator_class, "remove", Falcon::core::Iterator_remove );
   core->addClassMethod( iterator_class, "equal", Falcon::core::Iterator_equal );
   core->addClassMethod( iterator_class, "clone", Falcon::core::Iterator_clone );
   core->addClassProperty( iterator_class, "origin" );
   core->addClassProperty( iterator_class, "_pos" );

   return core;
}

}

/* end of core_func.cpp */
