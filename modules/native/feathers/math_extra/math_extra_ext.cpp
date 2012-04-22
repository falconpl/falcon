/* FALCON - The Falcon Programming Language.
 * FILE: math_extra_ext.cpp
 * 
 * Extra math functions
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

/** \file
  Extra math functions
  Interface extension functions
  */

/*#
   @beginmodule feathers.math_extra
*/

#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/matherror.h>

#include "math_extra_ext.h"
#include "math_extra_mod.h"

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
          throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
      }
      if ( res == 0.0 )
      {
         throw new MathError( ErrorParam( e_div_by_zero, __LINE__).origin( ErrorParam::e_orig_runtime ) );
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
        FALCON_DEFINE_FUNCTION_P(cosh)
        {
            Item *num1 = ctx->param( 0 );         
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
               throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = cosh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(sinh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = sinh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(tanh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = tanh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(acosh)
        {
            Item *num1 = ctx->param( 0 );         
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = acosh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(asinh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = asinh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(atanh)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = atanh( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(lambda)
        {
            Item *num1 = ctx->param( 0 );
            Item *num2 = ctx->param( 1 );

            if ( ! num1->isOrdinal() || ! num2->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = pow( num1->forceNumeric(), pow( num1->forceNumeric(), (num2->forceNumeric())-1 ) ) ;
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 2, res );
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
        FALCON_DEFINE_FUNCTION_P(sec)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = 1 / cos( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
            }
        }
        
        /*#
          @function csc
          @brief Returns the cosecant of the argument.
          @return The coseant ( 1 / sin() ) of the argument.
          @raise MathError If the argument is out of domain.

          The function may raise an error if the value cannot
          be computed because of a domain or overflow error.
          */
        FALCON_DEFINE_FUNCTION_P(cosec)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = 1 / sin( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
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
        FALCON_DEFINE_FUNCTION_P(cotan)
        {
            Item *num1 = ctx->param( 0 );
            if ( num1 == 0 || ! num1->isOrdinal() )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( ErrorParam::e_orig_runtime ).extra("N") );
            }

            errno = 0;
            numeric res = 1 / tan( num1->forceNumeric() );
            if ( errno != 0 )
            {
                throw new MathError( ErrorParam( e_domain, __LINE__).origin( ErrorParam::e_orig_runtime ) );
            }
            else {                 
                ctx->stackResult( 1, res );
            }
        }
        
    }
} // namespace Falcon::Ext

/* end of math_extra_ext.cpp */

