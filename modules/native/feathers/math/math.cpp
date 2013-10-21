/* FALCON - The Falcon Programming Language.
 * FILE: math_extra.cpp
 * 
 * Extra math functions
 * Main module file, providing the module object to the Falcon engine.
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
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include "math_ext.h"

/*#
   @module feathers.math
   @brief Mathematical functions

   The @b math module provides standard mathematical function that
   operate on IEEE 64 bit double-precision floating point numbers.
   
   Functions available in this module cover:
   - Trigonometry;
   - Logarithms and exponentials;
   - Floating point manipulation;
   - Combinatory calculus.
*/

//Define the math_extra module class
class MathExtraModule: public Falcon::Module
{
public:
   // initialize the module
   MathExtraModule():
      Module("math")
   {
      //============================================================
      // Api Declartion
      //
      
      // Hyperbolic
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(cosh) );
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(sinh) );
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(tanh) );
      
      // Inverse Hyperbolic
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(acosh) );
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(asinh) );
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(atanh) );

      // Reciprocal trigonometric function
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(sec) );
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(cosec) );
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(cotan) );

      // Other
      addMantra( new Falcon::Ext::FALCON_FUNCTION_NAME(lambda) );

      // Standard
      *this
         << new Falcon::Ext::FALCON_FUNCTION_NAME(log)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(log10)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(log2)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(logN)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(exp)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(sqrt)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(mod)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(pow)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(sin)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(cos)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(tan)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(asin)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(acos)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(atan)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(atan2)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(rad2deg)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(deg2rad)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(fract)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(fint)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(round)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(floor)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(ceil)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(abs)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(factorial)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(permutations)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(combinations)
               ;

   }
   virtual ~MathExtraModule() {}
};

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new MathExtraModule;
   return mod;
}

/* end of math_extra.cpp */

