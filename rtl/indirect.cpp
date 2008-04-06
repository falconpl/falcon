/*
   FALCON - The Falcon Programming Language
   FILE: indirect.cpp

   Indirect function calling and symbol access.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio apr 13 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include "rtl_messages.h"

/** \file
   Indirect function calling and symbol access.
*/

namespace Falcon {
namespace ext {

/**
   call( callable [, arrayOfParams] )
*/

FALCON_FUNC  call( ::Falcon::VMachine *vm )
{
   Item *func_x = vm->param(0);
   Item *params_x = vm->param(1);

   if ( func_x == 0 || ! func_x->isCallable() ||
       ( params_x != 0 && ! params_x->isArray() ) )
       {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( "C,A" ) ) );
      return;
   }

   // fetch the item here, as we're going to change the vector.
   Item func = *func_x;

   uint32 count = 0;

   if ( params_x != 0 )
   {
      CoreArray *array = params_x->asArray();
      count = array->length();
      for( uint32 i = 0; i < count; i++ )
      {
         vm->pushParameter( (*array)[ i ] );
      }
   }

   vm->callFrame( func, count );
}

/**
   methodCall( object, method, [, arrayOfParams] )
*/

FALCON_FUNC  methodCall( ::Falcon::VMachine *vm )
{
   Item *obj_x = vm->param(0);
   Item *method_x = vm->param(1);
   Item *params_x = vm->param(2);

   if ( obj_x == 0 || ! obj_x->isObject() || method_x == 0 || ! method_x->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            //"requres an object and a string representing a method" );
      return;
   }

   if ( params_x != 0 && ! params_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         //"optional third parameter must be an array" );
      return;
   }

   Item method;
   CoreObject *self = obj_x->asObject();
   if ( ! self->getProperty( *method_x->asString(), method ) ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( ! method.isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int count = 0;
   if ( params_x != 0 )
   {
      CoreArray *array = params_x->asArray();
      Item *elements = array->elements();
      count = array->length();
      for ( int i = 0; i < count; i ++ ) {
         vm->pushParameter( elements[ i ] );
      }
   }

   method.methodize( self );
   vm->callFrame( method, count );
}


static void internal_marshal( VMachine *vm, Item *message, Item *prefix, Item *if_not_found,
   const char *func_format )
{
  if ( ! vm->sender().isObject() && ! vm->self().isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString(msg::rtl_sender_not_object ) ) ) );
      return;
   }

   if ( message == 0 ||  ! message->isArray() ||
         ( prefix != 0 && ! prefix->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( func_format ) ) );
      return;
   }

   CoreArray &amsg = *message->asArray();
   if( amsg.length() == 0 || ! amsg[0].isString() || amsg[0].asString()->size() == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_marshall_not_cb ) ) ) );
      return;
   }

   // Ok, we're clear.
   // should we add a prefix to the marshalled element?
   String *method_name;
   if ( prefix != 0 && prefix->asString()->size() > 0 )
   {
      method_name = new GarbageString( vm, *amsg[0].asString() );
      const String &temp = *prefix->asString();
      // is the last character a single quote?
      if ( temp.size() > 0 )
      {
         if( temp.getCharAt( temp.length() - 1 ) == '\'' )
         {
            uint32 cFront = method_name->getCharAt(0);
            if ( cFront >= 'a' && cFront <= 'z' )
               method_name->setCharAt( 0, cFront - 'a' + 'A' );

            method_name->prepend( temp.subString(0, temp.length() - 1 ) );
         }
         else
            method_name->prepend( temp );
      }
   }
   else
      method_name = amsg[0].asString();


   // do the marshalled method exist and is it callable?
   Item method;
   CoreObject *self = vm->self().isObject() ? vm->self().asObject() : vm->sender().asObject();
   if ( ! self->getProperty( *method_name, method ) ||
        ! method.isCallable() )
   {
      // if not, call the item
      if ( if_not_found == 0 )
      {
         vm->raiseModError( new RangeError( ErrorParam( e_non_callable, __LINE__ ).
            origin(e_orig_runtime) ) );
         return;
      }
      else
         vm->retval( *if_not_found );
      return;
   }

   int count = 0;
   count = amsg.length();
   for ( int i = 1; i < count; i ++ ) {
      vm->pushParameter( amsg[ i ] );
   }

   method.methodize( self );
   vm->callFrame( method, count-1 );
}


FALCON_FUNC  marshalCB( ::Falcon::VMachine *vm )
{
   Item *message = vm->param(0);
   Item *prefix = vm->param(1);
   Item *if_not_found = vm->param(2);

   internal_marshal( vm, message, prefix, if_not_found, "A,[S,X]" );
}

FALCON_FUNC  marshalCBX( ::Falcon::VMachine *vm )
{
   Item *prefix = vm->param(0);
   Item *if_not_found = vm->param(1);
   Item *message = vm->param(2);

   internal_marshal( vm, message, prefix, if_not_found, "S,X,A" );
}

FALCON_FUNC  marshalCBR( ::Falcon::VMachine *vm )
{
   Item *prefix = vm->param(0);
   Item *message = vm->param(1);

   internal_marshal( vm, message, prefix, 0, "S,A" );
}




}
}

/* end of indirect.cpp */
