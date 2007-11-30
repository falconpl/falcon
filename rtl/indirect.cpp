/*
   FALCON - The Falcon Programming Language
   FILE: indirect.cpp
   $Id: indirect.cpp,v 1.6 2007/03/04 17:39:03 jonnymind Exp $

   Indirect function calling and symbol access.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio apr 13 2006
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
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>

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

   vm->callFrame( method, count );
}

}
}

/* end of indirect.cpp */
