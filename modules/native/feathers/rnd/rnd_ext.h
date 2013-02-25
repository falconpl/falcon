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
         FALCON_DECLARE_FUNCTION(cosh,"arg:N")
         FALCON_DECLARE_FUNCTION(sinh,"arg:N")
         FALCON_DECLARE_FUNCTION(tanh,"arg:N")

         // Inverse Hyperbolic
         FALCON_DECLARE_FUNCTION(acosh,"arg:N")
         FALCON_DECLARE_FUNCTION(asinh,"arg:N")
         FALCON_DECLARE_FUNCTION(atanh,"arg:N")

         // Other
         FALCON_DECLARE_FUNCTION(lambda,"num1:N,num2:N")

         // Reciprocal trigonometric function
         FALCON_DECLARE_FUNCTION(sec,"arg:N")
         FALCON_DECLARE_FUNCTION(cosec,"arg:N")
         FALCON_DECLARE_FUNCTION(cotan,"arg:N")

         // plain standard math
         FALCON_DECLARE_FUNCTION(log,"arg:N")
         FALCON_DECLARE_FUNCTION(log10,"arg:N")
         FALCON_DECLARE_FUNCTION(log2,"arg:N")
         FALCON_DECLARE_FUNCTION(logN,"arg:N, base:N")
         FALCON_DECLARE_FUNCTION(exp,"arg:N")
         FALCON_DECLARE_FUNCTION(sqrt,"arg:N")
         FALCON_DECLARE_FUNCTION(mod,"arg:N")
         FALCON_DECLARE_FUNCTION(pow,"arg:N")
         FALCON_DECLARE_FUNCTION(sin,"arg:N")
         FALCON_DECLARE_FUNCTION(cos,"arg:N")
         FALCON_DECLARE_FUNCTION(tan,"arg:N")
         FALCON_DECLARE_FUNCTION(asin,"arg:N")
         FALCON_DECLARE_FUNCTION(acos,"arg:N")
         FALCON_DECLARE_FUNCTION(atan,"arg:N")
         FALCON_DECLARE_FUNCTION(atan2,"arg:N")
         FALCON_DECLARE_FUNCTION(rad2deg,"arg:N")
         FALCON_DECLARE_FUNCTION(deg2rad,"arg:N")
         FALCON_DECLARE_FUNCTION(fract,"arg:N")
         FALCON_DECLARE_FUNCTION(fint,"arg:N")
         FALCON_DECLARE_FUNCTION(round,"arg:N")
         FALCON_DECLARE_FUNCTION(floor,"arg:N")
         FALCON_DECLARE_FUNCTION(ceil,"arg:N")
         FALCON_DECLARE_FUNCTION(abs,"arg:N")
         FALCON_DECLARE_FUNCTION(factorial,"arg:N")
         FALCON_DECLARE_FUNCTION(permutations, "n:N,p:N" )
         FALCON_DECLARE_FUNCTION(combinations, "n:N,p:N" )

         FALCON_DECLARE_FUNCTION( random, "..." )
         FALCON_DECLARE_FUNCTION( randomChoice, "first:X,second:X,...")
         FALCON_DECLARE_FUNCTION( randomPick, "series:A" )
         FALCON_DECLARE_FUNCTION( randomWalk, "series:A,size:[N]" )
         FALCON_DECLARE_FUNCTION( randomGrab, "series:A,size:[N]" )
         FALCON_DECLARE_FUNCTION( randomDice, "dices:N,sides:[N]" )
         FALCON_DECLARE_FUNCTION( randomSeed, "seed:[N]" )
    }
} // namespace Falcon::Ext

#endif

/* end of math_ext_ext.h */

