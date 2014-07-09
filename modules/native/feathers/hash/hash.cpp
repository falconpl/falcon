/*
   FALCON - The Falcon Programming Language.
   FILE: hash_ext.cpp

   Provides multiple hashing algorithms
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Thu, 25 Mar 2010 02:46:10 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

#include "hash_fm.h"

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
    return new Falcon::Feathers::ModuleHash;
}

#endif

/* end of hash.cpp */
