/* FALCON - The Falcon Programming Language.
 * FILE: rnd_ext.h
 * 
 * Simple random functions
 * Interface extension functions
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 24 Feb 2013 22:37:06 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2013: The above AUTHOR
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

#ifndef FALCON_RND_H
#define FALCON_RND_H

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/types.h>
#include <falcon/error.h>
#include <falcon/function.h>

namespace Falcon { 
    namespace Ext {
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

/* end of rnd_ext.h */

