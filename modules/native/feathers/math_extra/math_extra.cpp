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
#include "math_extra_ext.h"

#include "version.h"

/*#
   @module feathers.math_extra Uncommon math functions
   @brief Uncommon math functions
   
   The @b math_extra module provides some mathematical functions that are not
   commonly used in scripting languages.
*/

//Define the math_extra module class
class MathExtraModule: public Falcon::Module
{
public:
   // initialize the module
   MathExtraModule():
      Module("math_extra")
   {

      //language( "en_US" );
      //engineVersion( FALCON_VERSION_NUM );
      //version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

      //============================================================
      // Api Declartion
      //
      
      // Hyperbolic
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(cosh) );
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(sinh) );
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(tanh) );
      
      // Inverse Hyperbolic
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(acosh) );
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(asinh) );
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(atanh) );

      // Reciprocal trigonometric function
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(sec) );
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(cosec) );
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(cotan) );

      // Other
      addFunction( new Falcon::Ext::FALCON_FUNCTION_NAME(lambda) );

   }
   virtual ~MathExtraModule() {}
};

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new MathExtraModule;
   return mod;
}

/* end of math_extra.cpp */

