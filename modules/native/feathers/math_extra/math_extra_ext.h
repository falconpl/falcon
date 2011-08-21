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
#include <falcon/function.h>

#include <cmath>
#include <cerrno>

namespace Falcon { 
    namespace Ext {
        // Hyperbolic
        FALCON_DECLARE_FUNCTION(cosh,"")
        FALCON_DECLARE_FUNCTION(sinh,"")
        FALCON_DECLARE_FUNCTION(tanh,"")

        // Inverse Hyperbolic
        FALCON_DECLARE_FUNCTION(acosh,"")
        FALCON_DECLARE_FUNCTION(asinh,"")
        FALCON_DECLARE_FUNCTION(atanh,"")

        // Other
        FALCON_DECLARE_FUNCTION(lambda,"")

        // Reciprocal trigonometric function
        FALCON_DECLARE_FUNCTION(sec,"")
        FALCON_DECLARE_FUNCTION(cosec,"")
        FALCON_DECLARE_FUNCTION(cotan,"")
    }
} // namespace Falcon::Ext

#endif

/* end of math_ext_ext.h */

