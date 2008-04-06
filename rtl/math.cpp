/*
   FALCON - The Falcon Programming Language
   FILE: math.cpp

   Mathematical basic function for basic rtl.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom apr 16 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Mathematical basic function for basic rtl.
*/

#include <falcon/module.h>
#include <falcon/vm.h>

#include <math.h>
#include <errno.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC flc_math_log( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = log( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "log()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_exp( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = exp( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "exp()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_pow( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = pow( num1->forceNumeric(), num2->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "pow()" );
   }
   else {
      vm->retval( res );
   }
}


FALCON_FUNC flc_math_sin( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = sin( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "pow()" );
   }
   else {
      vm->retval( res );
   }
}


FALCON_FUNC flc_math_cos( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = cos( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "cos()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_tan( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = tan( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "tan()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_asin( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = asin( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "asin()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_acos( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = acos( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "acos()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_atan( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = atan( num1->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "atan()" );
   }
   else {
      vm->retval( res );
   }
}

FALCON_FUNC flc_math_atan2( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 1 );

   if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   errno = 0;
   numeric res = atan2( num1->forceNumeric(), num2->forceNumeric() );
   if ( errno != 0 )
   {
      vm->raiseError( e_domain, "atan2()" );
   }
   else {
      vm->retval( res );
   }
}

#define PI 3.1415926535897932384626433832795

FALCON_FUNC flc_math_rad2deg( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( 180.0 / ( PI * num1->forceNumeric() ) );
}

FALCON_FUNC flc_math_deg2rad( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( num1->forceNumeric() * PI / 180.0 );
}


FALCON_FUNC  flc_fract ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      vm->retval( (int64) 0 );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
      numeric intpart;
      vm->retval( modf( num->asNumeric(), &intpart ) );
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}

FALCON_FUNC  flc_fint( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      vm->retval( *num );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
         numeric n = num->asNumeric();
         numeric intpart;
         modf(n, &intpart );
         vm->retval( intpart );
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}


FALCON_FUNC  flc_round ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      vm->retval( *num );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
      #ifdef _MSC_VER
         numeric n = num->asNumeric();
         numeric intpart;
         numeric fractpart = modf(n, &intpart );

         if ( fractpart >= 0.5 )
            vm->retval( intpart + 1 );
         else if ( fractpart <= -0.5 )
            vm->retval( intpart - 1 );
         else
            vm->retval( intpart );
      #else
         vm->retval( llround( num->asNumeric() ) );
      #endif
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}

FALCON_FUNC  flc_floor ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      vm->retval( *num );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
      vm->retval( (int64) floor( num->asNumeric() ) );
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}

FALCON_FUNC  flc_ceil ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      vm->retval( *num );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
      vm->retval( (int64) ceil( num->asNumeric() ) );
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}

FALCON_FUNC  flc_abs ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      int64 n = num->asInteger();
      vm->retval( n < 0 ? -n : n );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
      numeric n = num->asNumeric();
      vm->retval( fabs( n ) );
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}




}
}

/* end of math.cpp */
