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

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "math_extra" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Api Declartion
   //

   // Hyperbolic
   self->addExtFunc( "cosh",  Falcon::Ext::Func_cosh );
   self->addExtFunc( "sinh",  Falcon::Ext::Func_sinh );
   self->addExtFunc( "tanh",  Falcon::Ext::Func_tanh );

   // Inverse Hyperbolic
   self->addExtFunc( "acosh",  Falcon::Ext::Func_acosh );
   self->addExtFunc( "asinh",  Falcon::Ext::Func_asinh );
   self->addExtFunc( "atanh",  Falcon::Ext::Func_atanh );
   self->addExtFunc( "atan2",  Falcon::Ext::Func_atan2 );

   // Reciprocal trigonometric function
   self->addExtFunc( "sec",  Falcon::Ext::Func_sec );
   self->addExtFunc( "csc",  Falcon::Ext::Func_csc );
   self->addExtFunc( "cotan",  Falcon::Ext::Func_cotan );

   // Other
   self->addExtFunc( "lambda", Falcon::Ext::Func_lambda );

   return self;
}

/* end of math_extra.cpp */

