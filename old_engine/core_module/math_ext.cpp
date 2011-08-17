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

/*#
   @beginmodule core
*/

#include <falcon/module.h>
#include <falcon/vm.h>

#include <math.h>
#include <errno.h>


/*#
   @funset core_math Math functions.
   @brief Functions providing math support to Falcon.

   This group includes mathematical, trigonometrical and floating point conversion
   functions.

   @beginset core_math
*/

namespace Falcon {
namespace core {

/*#
   @function log
   @brief Returns the natural logarithm of the argument.
   @param x Argument.
   @return The natural logarithm of the argument.
   @raise MathError If the argument is out of domain.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_log( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }

   errno = 0;
   numeric res = log( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function log10
   @brief Returns the common (base 10) logarithm of the argument.
   @param x Argument.
   @return The common logarithm of the argument.
   @raise MathError If the argument is out of domain.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_log10( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }

   errno = 0;
   numeric res = log10( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function exp
   @brief Returns exponential (e^x) of the argument.
   @param x Argument.
   @return The exponential of the argument.
   @raise MathError If the argument is out of domain.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_exp( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   errno = 0;
   numeric res = exp( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function sqrt
   @brief Returns the square root of the argument.
   @param x Argument.
   @return The square root of the argument.
   @raise MathError If the argument is out of domain.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_sqrt( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   errno = 0;
   numeric res = sqrt( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function mod
   @brief Returns the modulo of two arguments.
   @param x Argument.
   @param y Argument.
   @return The modulo of the two argument; x mod y.
   @raise MathError If the argument is out of domain.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_mod( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 1 );

   if ( num2 == 0 || ! num1->isOrdinal() || ! num2->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N,N") );
      return;
   }

   errno = 0;
   numeric res = fmod( num1->forceNumeric(), num2->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function pow
   @brief Returns the first argument elevated to the second one (x^y)
   @param x Base.
   @param y Exponent.
   @return x^y
   @raise MathError If the argument is out of domain.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_pow( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 1 );

   if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N,N") );
      return;
   }

   errno = 0;
   numeric res = pow( num1->forceNumeric(), num2->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function sin
   @brief Returns the sine of the argument.
   @param x Argument.
   @return The sine of the argument.
   @raise MathError If the argument is out of domain.

   The return value is expressed in radians.

   The function may raise an error if the value cannot be computed
   because of domain or overflow errors.
*/
FALCON_FUNC flc_math_sin( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }

   errno = 0;
   numeric res = sin( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function cos
   @brief Returns the cosine of the argument.
   @param x Argument.
   @return The cosine of the argument.
   @raise MathError If the argument is out of domain.

   The return value is expressed in radians.

   The function may raise an error if the value cannot be computed
   because of domain or overflow errors.
*/
FALCON_FUNC flc_math_cos( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra( "N" ) );
   }

   errno = 0;
   numeric res = cos( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function tan
   @brief Returns the tangent of the argument.
   @param x Argument.
   @return The tangent of the argument.
   @raise MathError If the argument is out of domain.

   The return value is expressed in radians.

   The function may raise an error if the value cannot be computed
   because of domain or overflow errors.
*/
FALCON_FUNC flc_math_tan( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }

   errno = 0;
   numeric res = tan( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function asin
   @brief Returns the arc sine of the argument.
   @param x Argument.
   @return The arc sine of the argument.
   @raise MathError If the argument is out of domain.

   The return value is expressed in radians.

   The function may raise an error if the value cannot be
   computed because of domain or overflow errors.
*/

FALCON_FUNC flc_math_asin( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }

   errno = 0;
   numeric res = asin( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function acos
   @brief Returns the arc cosine of the argument.
   @param x Argument.
   @return The arc cosine of the argument.
   @raise MathError If the argument is out of domain.

   This function computes the principal value of the arc cosine
   of its argument x. The value of x should be in the range [-1,1].

   The return value is expressed in radians.

   The function may raise a Math error if the value cannot
   be computed because of domain or overflow errors.
*/

FALCON_FUNC flc_math_acos( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   errno = 0;
   numeric res = acos( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function atan
   @brief Returns the arc tangent of the argument.
   @param x Argument.
   @return The arc tangent of the argument.
   @raise MathError If the argument is out of domain.

   This function computes the principal value of the arc tangent
   of its argument x. The value of x should be in the range [-1,1].

   The return value is expressed in radians.

   The function may raise a Math error if the value cannot
   be computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_atan( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   errno = 0;
   numeric res = atan( num1->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function atan2
   @brief Returns the arc tangent of x / y.
   @param x First argument.
   @param y Second argument.
   @return The arc tangent of the x / y.
   @raise MathError If the argument is out of domain.

   This function computes the principal value of the arc
   tangent of x/y, using the signs of both arguments to
   determine the quadrant of the return value.

   The return value is expressed in radians.

   The function may raise a Math error if the value cannot
   be computed because of domain or overflow errors.
*/
FALCON_FUNC flc_math_atan2( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 1 );

   if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   errno = 0;
   numeric res = atan2( num1->forceNumeric(), num2->forceNumeric() );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

#define PI 3.1415926535897932384626433832795
#define E  2.7182818284590452353602874713527
/*#
   @function rad2deg
   @brief Converts an angle expressed in radians into degrees.
   @param x An angle expressed in radians.
   @return The angle converted in degrees.
*/
FALCON_FUNC flc_math_rad2deg( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   vm->retval( 180.0 / ( PI * num1->forceNumeric() ) );
}

/*#
   @function deg2rad
   @brief Converts an angle expressed in degrees into radians.
   @param x An angle expressed in degrees.
   @return The angle converted in radians.
*/
FALCON_FUNC flc_math_deg2rad( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      return;
   }

   vm->retval( num1->forceNumeric() * PI / 180.0 );
}


/*#
   @function fract
   @brief Returns the fractional part of a number.
   @param x Argument.
   @return The fractional part of a number.

   This function returns the non-integer part of a number.
   For example,
   @code
   > fract( 1.234 )
   @endcode

   would print 0.234.
*/

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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }
}

/*#
   @function fint
   @brief Returns the integer part of a floating point number as a floating point number.
   @param x Argument.
   @return A floating point number with fractional part zeroed.

   Fint function works like the core @a int function,
   but it returns a floating point number. For example,
   @b fint applied on 3.58e200 will return the same number,
   while @a int would raise a math error, as the number
   cannot be represented in a integer
   number that can store numbers up to +-2^63.

*/
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }
}

/*#
   @function round
   @brief Rounds a floating point to the nearest integer.
   @param x Argument.
   @return Nearest integer to x.

   Round returns the nearest integer value of a given
   floating point number. If the fractional part of the number
   is greater or equal to 0.5, the number is rounded up to the nearest
   biggest integer in absolute value, while if it's less than 0.5
   the number is rounded down to the mere integer part. For example, 1.6
   is rounded to 2, -1.6 is rounded to -2, 1.2 is rounded to 1
   and -1.2 is rounded to -1.
*/

FALCON_FUNC  flc_round ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   if ( num->type() == FLC_ITEM_INT )
   {
      vm->retval( *num );
   }
   else if ( num->type() == FLC_ITEM_NUM )
   {
      // Or windows or solaris, use a simple round trick.
      #if defined(_MSC_VER) || ( defined (__SVR4) && defined (__sun) )
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }
}

/*#
   @function floor
   @brief Returns the smallest integer near to the given value.
   @param x Argument.
   @return The smallest integer near to the given value.

   Floor function returns the smallest integer near to a given floating
   point number. For example, floor of 1.9 is 1, and floor of -1.9 is -2.
   If an integer number is given, then the function returns the same number.
   This is similar to fint(), but in case of negative numbers @a fint would
   return the integer part; in case of -1.9 it would return -1.
*/
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }
}

/*#
   @function ceil
   @brief Returns the greatest integer near to the given value.
   @param x Argument.
   @return The ceil value.

   Ceil function returns the highest integer near to a given floating point
   number. For example, ceil of 1.1 is 2, and ceil of -1.1 is -1. If an
   integer number is given, then the function returns the same number.
*/

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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }
}

/*#
   @function abs
   @brief Returns the absolute value of a number.
   @param x A number.
   @return The absolute value of the parameter.

   If the argument is an integer, then an integer is returned,
   otherwise the return value will be a floating point number.
*/

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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }
}


static numeric fact( numeric n )
{
   numeric res = 1.0;
   while( n > 0 ) {
      res *= n;
      n = n-1.0;
   }

   return res;
}

/*#
   @function factorial
   @brief Returns the factorial of the argument.
   @param x Argument.
   @return The factorial of the argument.

   The return value is expressed as a floating point value.

   @note For high values of @b x, the function may require
   exponential computational time and power.
*/
FALCON_FUNC flc_math_factorial( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );

   if ( num1 == 0 || ! num1->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
   }

   numeric num = num1->forceNumeric();

   if ( num < 0 )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ) );
   }

   errno = 0;
   numeric res = fact( num );
   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function permutations
   @brief Returns the permutation of the arguments.
   @param x First argument.
   @param y Second arguments.
   @return The permutation of the arguments.

   The return value is expressed as a floating point value.

   @note For high values of @b x, the function may require
   exponential computational time and power.
*/

FALCON_FUNC flc_math_permutations( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 1 );

   if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N,N") );
      return;
   }

   numeric n = num1->forceNumeric();
   numeric r = num2->forceNumeric();

   // n must be > 0, but r may be zero.
   if ( n <= 0 || r < 0)
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ) );
   }

   errno = 0;
   // check to make sure numbers aren't the same
   double res = 1.0;
   double from = r == 0 ? 1 : n - r + 1;
   while ( from <= n && errno == 0 )
   {
      res *= from;
      from += 1.0;
   }

   if ( errno != 0 )
   {
      throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
   }
   else {
      vm->retval( res );
   }
}

/*#
   @function combinations
   @brief Returns the combination of the arguments.
   @param x First argument.
   @param y Second arguments.
   @return The combination of the arguments.

   The return value is expressed as a floating point value.

   @note For high values of @b x, the function may require
   exponential computational time and power.
*/
FALCON_FUNC flc_math_combinations( ::Falcon::VMachine *vm )
{
   Item *num1 = vm->param( 0 );
   Item *num2 = vm->param( 1 );

   if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N,N") );
   }

   numeric n = num1->forceNumeric();
   numeric r = num2->forceNumeric();
   // check to make sure numbers aren't the same
   if ( n <= 0 || r < 0)
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ) );
   }

   if ( n == r )
   {
      vm->retval( n );
   }
   else
   {
      errno = 0;
      numeric res = fact( n ) / (fact( r ) * fact(n-r));
      if ( errno != 0 )
      {
         throw new MathError( ErrorParam( e_domain, __LINE__).origin( e_orig_runtime ) );
      }
      else {
         vm->retval( res );
      }
   }
}

}
}

/* end of math.cpp */
