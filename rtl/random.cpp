/*
   FALCON - The Falcon Programming Language.
   FILE: random.cpp

   Random number related functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun nov 8 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Random number generator related functions.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/carray.h>
#include <falcon/string.h>
#include <falcon/sys.h>


#include <stdlib.h>
#include <math.h>

namespace Falcon {

/**
   random() --> [0 , 1 [
   random( NUMBER ) --> [ 0, NUMBER ]
   random( NUM1, NUM2 ) --> [ NUM1, NUM2 ]
   random( Item1, ... ItemN ) --> one of Item1 ... ItemN
*/

FALCON_FUNC  flc_random ( ::Falcon::VMachine *vm )
{
   int32 pcount = vm->paramCount();
   Item *elem1, *elem2;

   switch( pcount )
   {
      case 0:
         vm->retval( (numeric) rand() / ((numeric) RAND_MAX + 1e-64) );
      break;

      case 1:
         elem1 = vm->param(0);
         if ( elem1->isOrdinal() ) {
            int64 num = elem1->forceInteger() + 1;
            if ( num < 0 )
               vm->retval( -(((int64) rand()) % -num) );
            else if ( num == 0 )
               vm->retval( 0 );
            else
               vm->retval( ((int64) rand()) % num );
         }
         else
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      break;

      case 2:
         elem1 = vm->param(0);
         elem2 = vm->param(1);
         if ( elem1->isOrdinal() && elem2->isOrdinal() )
         {
            int64 num1 = elem1->forceInteger();
            int64 num2 = elem2->forceInteger();
            if ( num1 == num2 )
               vm->retval( num1 );
            else if ( num2 < num1 ) {
               int64 temp = num2;
               num2 = num1;
               num1 = temp;
            }
            num2 ++;

            vm->retval( num1 + ((int64) rand()) % (num2 - num1) );
         }
         else
            vm->retval( *vm->param( (rand() % 2) ) );
      break;

      default:
         vm->retval( *vm->param( rand() % pcount ) );
   }
}

FALCON_FUNC  flc_randomChoice( ::Falcon::VMachine *vm )
{
   int32 pcount = vm->paramCount();

   switch( pcount )
   {
      case 0:
      case 1:
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      break;

      default:
         vm->retval( *vm->param( rand() % pcount ) );
   }
}


FALCON_FUNC  flc_randomPick ( ::Falcon::VMachine *vm )
{
   Item *series = vm->param(0);

   if ( series == 0 || ! series->isArray() || series->asArray()->length() == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreArray &source = *series->asArray();
   vm->retval( source[ rand() % source.length() ] );
}


FALCON_FUNC  flc_randomWalk ( ::Falcon::VMachine *vm )
{
   Item *series = vm->param(0);
   Item *qty = vm->param(1);

   if ( series == 0 || ! series->isArray() )  {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   if ( qty != 0 && ! qty->isOrdinal() )  {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 number = qty == 0 ? 1 : (int32)qty->forceInteger();
   if( number < 1 ) number = 1;

   CoreArray *array = new CoreArray( vm, number );
   CoreArray &source = *series->asArray();
   int32 slen = (int32) source.length();

   if ( slen > 0 ) {
      while( number > 0 ) {
         array->append( source[ rand() % slen ] );
         number--;
      }
   }

   vm->retval( array );
}


FALCON_FUNC  flc_randomGrab ( ::Falcon::VMachine *vm )
{
   Item *series = vm->param(0);
   Item *qty = vm->param(1);

   if ( series == 0 || ! series->isArray() )  {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   if ( qty != 0 && ! qty->isOrdinal() )  {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 number = qty == 0 ? 1 : (int32)qty->forceInteger();
   if( number < 1 ) number = 1;

   CoreArray *array = new CoreArray( vm, number );
   CoreArray &source = *series->asArray();
   int32 slen = (int32) source.length();

   while( number > 0 && slen > 0 ) {
      uint32 pos = rand() % slen;
      array->append( source[ pos ] );
      source.remove( pos );
      slen--;
      number--;
   }

   vm->retval( array );
}


FALCON_FUNC  flc_randomSeed ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   unsigned int value;

   if ( num == 0 )
   {
      value = (unsigned int) (Sys::_seconds() * 1000);
   }
   else {
      if ( ! num->isOrdinal() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      value = (unsigned int) num->forceInteger();
   }

   srand( value );
}

}


/* end of random.cpp */
