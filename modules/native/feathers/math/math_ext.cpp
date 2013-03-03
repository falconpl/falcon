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

#include "math_ext.h"

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
    }
} // namespace Falcon::Ext

/* end of math_ext.cpp */

