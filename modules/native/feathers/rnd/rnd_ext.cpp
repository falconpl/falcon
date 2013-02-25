/* FALCON - The Falcon Programming Language.
 * FILE: math_ext.cpp
 * 
 * Extra functions
 * Interface extension functions
 * -------------------------------------------------------------------
 * Author: Steven N Oliver
 * Begin: Wed, 27 Oct 2010 20:12:51 -0400
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2010: The above AUTHOR
 * 
 * Licensed under the Falcon Programming Language License,
 * Version 1.1 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain
 * a copy of the License at
 * 
 * http://www.falconpl.org/?page_id=license_1_1
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#undef SRC
#define SRC "modules/native/feathers/math/math_ext.cpp"

/** \file
  Extra math functions
  Interface extension functions
  */

/*#
   @beginmodule feathers.math
*/

#include <falcon/engine.h>
#include <falcon/vmcontext.h>

#include <falcon/error.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/matherror.h>

// random
#include <falcon/vm.h>
#include <falcon/itemarray.h>
#include <falcon/sys.h>

#include "math_ext.h"
#include "math_mod.h"

namespace Falcon { 
    namespace Ext {

// visual studio doesn't have inverse hyperbolic functions
#ifdef _MSC_VER
   static double __inverse_call( double value, double (*func)(double) )
   {      
      errno = 0;
      double res = func( value );
      if ( errno != 0 )
      {
          throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
      }
      if ( res == 0.0 )
      {
         throw new MathError( ErrorParam( e_div_by_zero, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
      }
      return 1/res;
   }

   static double acosh( double value ) { return __inverse_call( value, cosh ); }
   static double asinh( double value ) { return __inverse_call( value, sinh ); }
   static double atanh( double value ) { return __inverse_call( value, tanh ); }
#endif

        // Hyperbolic          
        /*#
          @function cosh
          @brief Returns the hyperbolic cosine of the argument.
          @return The hyperbolic cosine of the argument.
          @raise MathError If the argument is out of domain.
          
          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(cosh)
        {
            Item *num1 = ctx->param( 0 );         
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
               throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = cosh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function sinh
          @brief Returns the hyperbolic sine of the argument.
          @return The hyperbolic sine of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(sinh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = sinh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function tanh
          @brief Returns the hyperbolic tangent of the argument.
          @return The hyperbolic tangent of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(tanh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = tanh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }

        // Inverse Hyperbolic
        /*#
          @function acosh
          @brief Returns the hyperbolic cosine of the argument.
          @return The inverse hyperbolic cosine of the argument.
          @raise MathError If the argument is out of domain.
          
          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(acosh)
        {
            Item *num1 = ctx->param( 0 );         
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = acosh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function asinh
          @brief Returns the hyperbolic sine of the argument.
          @return The inverse hyperbolic sine of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(asinh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = asinh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function atanh
          @brief Returns the hyperbolic tangent of the argument.
          @return The inverse hyperbolic tangent of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(atanh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = atanh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }

        /*#
          @function lambda
          @brief Returns the lambda of two arguments.
          @return The lambda which is arg1 raised to itself, raised to arg2 - 1.
          @raise MathError If the value is to large.

          The function may raise an error if the value cannot
          be computed because of an overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(lambda)
        {
            Item *num1 = ctx->param( 0 );
            Item *num2 = ctx->param( 1 );

            if ( ! num1->isOrdinal() || ! num2->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = pow( num1->forceNumeric(), pow( num1->forceNumeric(), (num2->forceNumeric())-1 ) ) ;
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function sec
          @brief Returns the secant of the argument.
          @return The reciprocal cosine ( 1 / cos() ) of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(sec)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = 1 / cos( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function cosec
          @brief Returns the cosecant of the argument.
          @return The coseant ( 1 / sin() ) of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(cosec)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = 1 / sin( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        
        /*#
          @function cotan
          @brief Returns the cotangent of the argument.
          @return The reciprocal of the tangent ( 1 / tan() ) of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P1(cotan)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = 1 / tan( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__, SRC).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->returnFrame( res );
            }
        }
        

        /*#
           @function log
           @brief Returns the natural logarithm of the argument.
           @param arg Argument.
           @return The natural logarithm of the argument.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(log)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           errno = 0;
           numeric res = log( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function log2
           @brief Returns the logarithm in base 2 of the argument.
           @param arg Argument.
           @return The logarithm in base 2 of the argument.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(log2)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           errno = 0;
           numeric res = log2( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function logN
           @brief Returns the logarithm in base N of the argument.
           @param arg Argument.
           @param base Base for the logarithm.
           @return The logarithm in base 2 of the argument.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(logN)
        {
           Item *num1 = ctx->param( 0 );
           Item *i_base = ctx->param( 1 );

           if ( num1 == 0 || ! num1->isOrdinal()
                    || i_base == 0 || ! i_base->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N,N") );
           }

           numeric base = i_base->forceNumeric();
           if( base == 0.0 )
           {
              throw new MathError( ErrorParam( e_div_by_zero, __LINE__, SRC ).extra("Logarithm base 0") );
           }

           errno = 0;
           numeric res = log( num1->forceNumeric() )/log(base);
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function log10
           @brief Returns the common (base 10) logarithm of the argument.
           @param arg Argument.
           @return The common logarithm of the argument.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(log10)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           errno = 0;
           numeric res = log10( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function exp
           @brief Returns exponential (e^x) of the argument.
           @param arg Argument.
           @return The exponential of the argument.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(exp)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           errno = 0;
           numeric res = exp( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function sqrt
           @brief Returns the square root of the argument.
           @param arg Argument.
           @return The square root of the argument.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(sqrt)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           errno = 0;
           numeric res = sqrt( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function mod
           @brief Returns the modulo of two arguments.
           @param arg Argument.
           @param y Argument.
           @return The modulo of the two argument; x mod y.
           @raise MathError If the argument is out of domain.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(mod)
        {
           Item *num1 = ctx->param( 0 );
           Item *num2 = ctx->param( 1 );

           if ( num2 == 0 || ! num1->isOrdinal() || ! num2->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N,N") );
              return;
           }

           errno = 0;
           numeric res = fmod( num1->forceNumeric(), num2->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
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
        FALCON_DEFINE_FUNCTION_P1(pow)
        {
           Item *num1 = ctx->param( 0 );
           Item *num2 = ctx->param( 1 );

           if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N,N") );
              return;
           }

           errno = 0;
           numeric res = pow( num1->forceNumeric(), num2->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function sin
           @brief Returns the sine of the argument.
           @param arg Argument.
           @return The sine of the argument.
           @raise MathError If the argument is out of domain.

           The return value is expressed in radians.

           The function may raise an error if the value cannot be computed
           because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(sin)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           errno = 0;
           numeric res = sin( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function cos
           @brief Returns the cosine of the argument.
           @param arg Argument.
           @return The cosine of the argument.
           @raise MathError If the argument is out of domain.

           The return value is expressed in radians.

           The function may raise an error if the value cannot be computed
           because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(cos)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra( "N" ) );
           }

           errno = 0;
           numeric res = cos( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function tan
           @brief Returns the tangent of the argument.
           @param arg Argument.
           @return The tangent of the argument.
           @raise MathError If the argument is out of domain.

           The return value is expressed in radians.

           The function may raise an error if the value cannot be computed
           because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(tan)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           errno = 0;
           numeric res = tan( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function asin
           @brief Returns the arc sine of the argument.
           @param arg Argument.
           @return The arc sine of the argument.
           @raise MathError If the argument is out of domain.

           The return value is expressed in radians.

           The function may raise an error if the value cannot be
           computed because of domain or overflow errors.
        */

        FALCON_DEFINE_FUNCTION_P1(asin)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           errno = 0;
           numeric res = asin( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function acos
           @brief Returns the arc cosine of the argument.
           @param arg Argument.
           @return The arc cosine of the argument.
           @raise MathError If the argument is out of domain.

           This function computes the principal value of the arc cosine
           of its argument x. The value of x should be in the range [-1,1].

           The return value is expressed in radians.

           The function may raise a Math error if the value cannot
           be computed because of domain or overflow errors.
        */

        FALCON_DEFINE_FUNCTION_P1(acos)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           errno = 0;
           numeric res = acos( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function atan
           @brief Returns the arc tangent of the argument.
           @param arg Argument.
           @return The arc tangent of the argument.
           @raise MathError If the argument is out of domain.

           This function computes the principal value of the arc tangent
           of its argument x. The value of x should be in the range [-1,1].

           The return value is expressed in radians.

           The function may raise a Math error if the value cannot
           be computed because of domain or overflow errors.
        */
        FALCON_DEFINE_FUNCTION_P1(atan)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           errno = 0;
           numeric res = atan( num1->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
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
        FALCON_DEFINE_FUNCTION_P1(atan2)
        {
           Item *num1 = ctx->param( 0 );
           Item *num2 = ctx->param( 1 );

           if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           errno = 0;
           numeric res = atan2( num1->forceNumeric(), num2->forceNumeric() );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
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
        FALCON_DEFINE_FUNCTION_P1(rad2deg)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           ctx->returnFrame( 180.0 / ( PI * num1->forceNumeric() ) );
        }

        /*#
           @function deg2rad
           @brief Converts an angle expressed in degrees into radians.
           @param x An angle expressed in degrees.
           @return The angle converted in radians.
        */
        FALCON_DEFINE_FUNCTION_P1(deg2rad)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
              return;
           }

           ctx->returnFrame( num1->forceNumeric() * PI / 180.0 );
        }


        /*#
           @function fract
           @brief Returns the fractional part of a number.
           @param arg Argument.
           @return The fractional part of a number.

           This function returns the non-integer part of a number.
           For example,
           @code
           > fract( 1.234 )
           @endcode

           would print 0.234.
        */

        FALCON_DEFINE_FUNCTION_P1(fract)
        {
           Item *num = ctx->param( 0 );
           if ( num->type() == FLC_ITEM_INT )
           {
              ctx->returnFrame( (int64) 0 );
           }
           else if ( num->type() == FLC_ITEM_NUM )
           {
              numeric intpart;
              ctx->returnFrame( modf( num->asNumeric(), &intpart ) );
           }
           else {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }
        }

        /*#
           @function fint
           @brief Returns the integer part of a floating point number as a floating point number.
           @param arg Argument.
           @return A floating point number with fractional part zeroed.

           Fint function works like the core @a int function,
           but it returns a floating point number. For example,
           @b fint applied on 3.58e200 will return the same number,
           while @a int would raise a math error, as the number
           cannot be represented in a integer
           number that can store numbers up to +-2^63.

        */
        FALCON_DEFINE_FUNCTION_P1(fint)
        {
           Item *num = ctx->param( 0 );
           if ( num->type() == FLC_ITEM_INT )
           {
              ctx->returnFrame( *num );
           }
           else if ( num->type() == FLC_ITEM_NUM )
           {
                 numeric n = num->asNumeric();
                 numeric intpart;
                 modf(n, &intpart );
                 ctx->returnFrame( intpart );
           }
           else {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }
        }

        /*#
           @function round
           @brief Rounds a floating point to the nearest integer.
           @param arg Argument.
           @return Nearest integer to x.

           Round returns the nearest integer value of a given
           floating point number. If the fractional part of the number
           is greater or equal to 0.5, the number is rounded up to the nearest
           biggest integer in absolute value, while if it's less than 0.5
           the number is rounded down to the mere integer part. For example, 1.6
           is rounded to 2, -1.6 is rounded to -2, 1.2 is rounded to 1
           and -1.2 is rounded to -1.
        */

        FALCON_DEFINE_FUNCTION_P1(round)
        {
           Item *num = ctx->param( 0 );
           if ( num->type() == FLC_ITEM_INT )
           {
              ctx->returnFrame( *num );
           }
           else if ( num->type() == FLC_ITEM_NUM )
           {
              // Or windows or solaris, use a simple round trick.
              #if defined(_MSC_VER) || ( defined (__SVR4) && defined (__sun) )
                 numeric n = num->asNumeric();
                 numeric intpart;
                 numeric fractpart = modf(n, &intpart );

                 if ( fractpart >= 0.5 )
                    ctx->returnFrame( intpart + 1 );
                 else if ( fractpart <= -0.5 )
                    ctx->returnFrame( intpart - 1 );
                 else
                    ctx->returnFrame( intpart );
              #else
                 ctx->returnFrame( llround( num->asNumeric() ) );
              #endif
           }
           else {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }
        }

        /*#
           @function floor
           @brief Returns the smallest integer near to the given value.
           @param arg Argument.
           @return The smallest integer near to the given value.

           Floor function returns the smallest integer near to a given floating
           point number. For example, floor of 1.9 is 1, and floor of -1.9 is -2.
           If an integer number is given, then the function returns the same number.
           This is similar to fint(), but in case of negative numbers @a fint would
           return the integer part; in case of -1.9 it would return -1.
        */
        FALCON_DEFINE_FUNCTION_P1(floor)
        {
           Item *num = ctx->param( 0 );
           if ( num->type() == FLC_ITEM_INT )
           {
              ctx->returnFrame( *num );
           }
           else if ( num->type() == FLC_ITEM_NUM )
           {
              ctx->returnFrame( (int64) floor( num->asNumeric() ) );
           }
           else {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }
        }

        /*#
           @function ceil
           @brief Returns the greatest integer near to the given value.
           @param arg Argument.
           @return The ceil value.

           Ceil function returns the highest integer near to a given floating point
           number. For example, ceil of 1.1 is 2, and ceil of -1.1 is -1. If an
           integer number is given, then the function returns the same number.
        */

        FALCON_DEFINE_FUNCTION_P1(ceil)
        {
           Item *num = ctx->param( 0 );
           if ( num->type() == FLC_ITEM_INT )
           {
              ctx->returnFrame( *num );
           }
           else if ( num->type() == FLC_ITEM_NUM )
           {
              ctx->returnFrame( (int64) ceil( num->asNumeric() ) );
           }
           else {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
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

        FALCON_DEFINE_FUNCTION_P1(abs)
        {
           Item *num = ctx->param( 0 );
           if ( num->type() == FLC_ITEM_INT )
           {
              int64 n = num->asInteger();
              ctx->returnFrame( n < 0 ? -n : n );
           }
           else if ( num->type() == FLC_ITEM_NUM )
           {
              numeric n = num->asNumeric();
              ctx->returnFrame( fabs( n ) );
           }
           else {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
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
           @param arg Argument.
           @return The factorial of the argument.

           The return value is expressed as a floating point value.

           @note For high values of @b x, the function may require
           exponential computational time and power.
        */
        FALCON_DEFINE_FUNCTION_P1(factorial)
        {
           Item *num1 = ctx->param( 0 );

           if ( num1 == 0 || ! num1->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N") );
           }

           numeric num = num1->forceNumeric();

           if ( num < 0 )
           {
              throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC ) );
           }

           errno = 0;
           numeric res = fact( num );
           if ( errno != 0 )
           {
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function permutations
           @brief Returns the permutation of the arguments.
           @param n First argument.
           @param p Second arguments.
           @return The permutation of the arguments.

           The return value is expressed as a floating point value.

           @note For high values of @b n, the function may require
           exponential computational time and power.
        */

        FALCON_DEFINE_FUNCTION_P1(permutations)
        {
           Item *num1 = ctx->param( 0 );
           Item *num2 = ctx->param( 1 );

           if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N,N") );
              return;
           }

           numeric n = num1->forceNumeric();
           numeric r = num2->forceNumeric();

           // n must be > 0, but r may be zero.
           if ( n <= 0 || r < 0)
           {
              throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC ) );
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
              throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
           }
           else {
              ctx->returnFrame( res );
           }
        }

        /*#
           @function combinations
           @brief Returns the combination of the arguments.
           @param n First argument.
           @param p Second arguments.
           @return The combination of the arguments.

           The return value is expressed as a floating point value.

           @note For high values of @b n, the function may require
           exponential computational time and power.
        */
        FALCON_DEFINE_FUNCTION_P1(combinations)
        {
           Item *num1 = ctx->param( 0 );
           Item *num2 = ctx->param( 1 );

           if ( num1 == 0 || ! num1->isOrdinal() || num2 == 0 || ! num2->isOrdinal() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC ).extra("N,N") );
           }

           numeric n = num1->forceNumeric();
           numeric r = num2->forceNumeric();
           // check to make sure numbers aren't the same
           if ( n <= 0 || r < 0)
           {
              throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC ) );
           }

           if ( n == r )
           {
              ctx->returnFrame( n );
           }
           else
           {
              errno = 0;
              numeric res = fact( n ) / (fact( r ) * fact(n-r));
              if ( errno != 0 )
              {
                 throw new MathError( ErrorParam( e_domain, __LINE__, SRC) );
              }
              else {
                 ctx->returnFrame( res );
              }
           }
        }


        /*#
           @function random
           @brief Returns a pseudo random number.
           @param ... See below.
           @return A pseudo random number or a random item picked from parameters.

           This function has actually several functionalities that are
           selected depending on the parameters.

           Without parameters, the function returns a floating point number in the
           range [0,1).

           With a signle numeric parameter, the function returns an integer between
           0 and the number, included. The following functions are equivalent:
           @code
              > random( x )
              > int( random() * (x + 1) )
           @endcode

           With two numeric parameters, the function returns an integer in the range
           [x, y]. The following functions are equivalent:
           @code
              > random( x, y )
              > int( x + (random() * (y + 1)) )
           @endcode

           With more than two parameters, or when at least one of the first two
           parameters it not a number, one of the parameter is picked at random
           and then returned.

           The function @a randomChoice returns unambiguously one of the parameters
           picked at random.

        */
        FALCON_DEFINE_FUNCTION_P( random )
        {
           Item *elem1, *elem2;
           MTRand_interlocked &rng = ctx->vm()->mtrand();

           switch( pCount )
           {
              case 0:
                 ctx->returnFrame( (numeric) rng.rand() );
              break;

              case 1:
                 elem1 = ctx->param(0);
                 if ( elem1->isOrdinal() ) {
                    int64 num = elem1->forceInteger();
                    if ( num < 0 )
                       ctx->returnFrame( -((int64) rng.randInt64( (-num) & ~(UI64LIT(1) << 63) )) ); // mask out sign bit and make result negative
                    else if ( num == 0 )
                       ctx->returnFrame( 0 );
                    else
                       ctx->returnFrame( (int64) rng.randInt64( num & ~(UI64LIT(1) << 63) ) ); // mask out sign bit, result always positive
                 }
                 else
                    throw paramError(__LINE__,SRC);
              break;

              case 2:
                 elem1 = ctx->param(0);
                 elem2 = ctx->param(1);
                 if ( elem1->isOrdinal() && elem2->isOrdinal() )
                 {
                    int64 num1 = elem1->forceInteger();
                    int64 num2 = elem2->forceInteger();
                    if ( num1 == num2 )
                       ctx->returnFrame( num1 );
                    else if ( num2 < num1 ) {
                       int64 temp = num2;
                       num2 = num1;
                       num1 = temp;
                    }
                    num2 ++;

                    ctx->returnFrame( (int64) (num1 + rng.randInt64(num2 - num1 - 1)) );
                 }
                 else
                    ctx->returnFrame( *ctx->param( rng.randInt(1) ) );
              break;

              default:
                 ctx->returnFrame( *ctx->param( rng.randInt(pCount - 1) ) );
                 break;
           }
        }

        /*#
           @function randomChoice
           @brief Selects one of the arguments at random and returns it.
           @param ... At least two items of any kind.
           @return One of the parameters, picked at random.

           This function works as @a random when it receives more than two
           parameters, but its usage is not ambiguous in case there are two
           items from which to choice. The function raises an error
           if less than two parameters are passed.
        */

        FALCON_DEFINE_FUNCTION_P(randomChoice)
        {
           switch( pCount )
           {
              case 0:
              case 1:
                 throw paramError(__LINE__,SRC);
              break;

              default:
              {
                 MTRand_interlocked &rng = ctx->vm()->mtrand();

                 ctx->returnFrame( *ctx->param( rng.randInt(pCount - 1) ) );
              }
              break;
           }
        }

        /*#
           @function randomPick
           @brief Grabs repeatedly random elements from an array.
           @param series An array containing one or more items.
           @return One of the items in the array.
           @raise ParamError if the @b series is empty.

           This function choices one of the items contained in the @b series array
           at random.

           If the array is empty, a ParamError error is raised.
        */
        FALCON_DEFINE_FUNCTION_P1( randomPick )
        {
           Item *series = ctx->param(0);
           if ( series == 0 || ! series->isArray() || series->asArray()->length() == 0 )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                    .extra( "A" ) );
           }

           MTRand_interlocked &rng = ctx->vm()->mtrand();

           ItemArray &source = *series->asArray();
           ctx->returnFrame( source[ rng.randInt(source.length() - 1) ] );
        }

        /*#
           @function randomWalk
           @brief Performs a random walk in an array.
           @param series An array containing one or more items.
           @optparam size Desire size of the walk.
           @return An array built from randomly picked items.

           This function picks one or more elements from the given array,
           and stores them in a new array without removing them from the
           old one. Elements can be picked up repeatedly, and the size
           of the target array may be larger than the size of the original one.

           If the requested target size is zero, or if the original array is empty,
           an empty array is returned.

           If @b size is not given, 1 is assumed; if it's less than zero,
           then an the function will create an array of the same size of the
           @b series array, but the target array can contain multiple copies
           of the items in @b series, or it may be missing some of them.
        */
        FALCON_DEFINE_FUNCTION_P1( randomWalk )
        {
           Item *series = ctx->param(0);
           Item *qty = ctx->param(1);

           if ( series == 0 || ! series->isArray()
              || (qty != 0 && ! qty->isOrdinal()) )
           {
              throw paramError(__LINE__,SRC);
           }

           MTRand_interlocked &rng =  ctx->vm()->mtrand();

           int32 number = qty == 0 ? 1 : (int32)qty->forceInteger();
           if( number < 0 ) number = series->asArray()->length();

           ItemArray *array = new ItemArray( number );
           ItemArray &source = *series->asArray();
           int32 slen = (int32) source.length();

           if ( slen > 0 ) {
              while( number > 0 ) {
                 array->append( source[ rng.randInt(slen - 1) ] );
                 number--;
              }
           }

           ctx->returnFrame( FALCON_GC_HANDLE(array) );
        }


        /*#
           @function randomGrab
           @brief Grabs repeatedly random elements from an array.
           @param series An array from which items will be extracted.
           @optparam size Count of extracted items.
           @return An array with some or all of the items grabbed from the original elements.

           This function extracts a desired amount of items from the elements array,
           putting them in a new array that will be returned. Items left in the elements
           array have a fair chance to be selected and removed at every step. If the size
           parameter is greater or equal than the size of the elements array, the array is
           eventually emptied and all the items are moved to the new array, actually
           performing a complete fair shuffling of the original.

           If @b size is not given, 1 is assumed; if it's zero or less than zero,
           then all the elements in the @b series array will be taken.

           This function is suitable to emulate card shuffling or other random
           extraction events.
        */

        FALCON_DEFINE_FUNCTION_P1( randomGrab )
        {
           Item *series = ctx->param(0);
           Item *qty = ctx->param(1);

           if ( series == 0 || ! series->isArray()
              || (qty != 0 && ! qty->isOrdinal()) )
           {
              throw paramError(__LINE__, SRC);
           }

           MTRand_interlocked &rng = ctx->vm()->mtrand();

           int32 number = qty == 0 ? 1 : (int32)qty->forceInteger();
           if( number < 1 ) number = series->asArray()->length();

           ItemArray *array = new ItemArray( number );
           ItemArray &source = *series->asArray();
           int32 slen = (int32) source.length();

           while( number > 0 && slen > 0 ) {
              uint32 pos = rng.randInt(slen - 1);
              array->append( source[ pos ] );
              source.remove( pos );
              slen--;
              number--;
           }

           ctx->returnFrame( FALCON_GC_HANDLE(array) );
        }

        /*#
           @function randomDice
           @brief Performs a virtual dice set trow.
           @param dices Number of dices to be thrown.
           @optparam sides Number of faces in the virtual dices.
           @return A random value which is the sum of the virtual throws.

           This function generates a series of successive @b dices throws,
           each one being integer value in the range [1, @b sides].

           If @b sides is not given, 6 is assumed.

           It would be easy to obtain the same result with simple instructions
           in Falcon, but this function spares several wasted VM cycles.

           The @b dices parameter must be greater than zero, and the
           and @b sides parameter must be greater than one.
        */

        FALCON_DEFINE_FUNCTION_P1( randomDice )
        {
           Item *i_dices = ctx->param(0);
           Item *i_sides = ctx->param(1);

           if ( i_dices == 0 || ! i_dices->isOrdinal() || ( i_sides != 0 && ! i_sides->isOrdinal()) )  {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
                 extra( "N,N") );
           }

           MTRand &rng = ctx->vm()->mtrand();

           int64 dices = i_dices->forceInteger();
           int64 sides = i_sides == 0 ? 6 : i_sides->forceInteger();
           if( dices < 1 || sides < 2 )
           {
              throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC ).
                 extra( "N>0,N>1" ) );
           }

           int64 result = 0;
           for( int64 i = 0; i < dices; i ++ )
           {
              result += 1 + rng.randInt64(sides - 1);
           }

           ctx->returnFrame( result );
        }

        /*#
           @function randomSeed
           @brief Seeds the random number generator.
           @optparam seed An integer number being used as random seed.

           The random seed should be set once per program, possibly using a number that
           is likely to vary greatly among different executions. A good seed may be the
           return of the seconds() function, eventually multiplied by 1000 to make the
           milliseconds to take part in the seeding. If called without parameters, a number
           based on the current system timer value will be used.

           Repeated calls to random(), and calls based on random function as
           randomChoice, randomPick and so on, will produce the same sequences if
           randomSeed() is called with the same seed.

           Using a constant number as random seed may be a good strategy to produce
           predictable debug sequences.
        */
        FALCON_DEFINE_FUNCTION_P1( randomSeed )
        {
           Item *num = ctx->param( 0 );
           uint32 value;

           if ( num == 0 )
           {
              value = (uint32) Sys::_milliseconds();
           }
           else
           {
              if ( ! num->isOrdinal() )
              {
                 throw paramError(__LINE__, SRC );
              }

              value = (uint32) num->forceInteger();
           }

           MTRand_interlocked &rng = ctx->vm()->mtrand();

           rng.seed( value );
        }


    }
} // namespace Falcon::Ext

/* end of math_ext.cpp */

