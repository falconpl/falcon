/* FALCON - The Falcon Programming Language.
 * FILE: math_extra_ext.h
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
  Interface extension functions - header file
  */

#ifndef math_extra_ext_H
#define math_extra_ext_H

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/types.h>
#include <falcon/error.h>

#include <cmath>
#include <cerrno>

namespace Falcon { 
    namespace Ext {
        // Hyperbolic
        FALCON_FUNC Func_cosh( ::Falcon::VMachine *vm );
        FALCON_FUNC Func_sinh( ::Falcon::VMachine *vm );
        FALCON_FUNC Func_tanh( ::Falcon::VMachine *vm );

        // Inverse Hyperbolic
        FALCON_FUNC Func_acosh( ::Falcon::VMachine *vm );
        FALCON_FUNC Func_asinh( ::Falcon::VMachine *vm );
        FALCON_FUNC Func_atanh( ::Falcon::VMachine *vm ); 
        FALCON_FUNC Func_atan2( ::Falcon::VMachine *vm ); 

        // Reciprocal trigonometric function
        FALCON_FUNC Func_sec( ::Falcon::VMachine *vm );
        FALCON_FUNC Func_csc( ::Falcon::VMachine *vm );
        FALCON_FUNC Func_cotan( ::Falcon::VMachine *vm ); 

        // Other
        FALCON_FUNC Func_lambda( ::Falcon::VMachine *vm );
    }
} // namespace Falcon::Ext

#endif

/* end of math_ext_ext.h */

