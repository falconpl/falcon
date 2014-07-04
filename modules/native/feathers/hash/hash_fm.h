/*
   FALCON - The Falcon Programming Language.
   FILE: hash_fm.h

   Provides multiple hashing algorithms
   Interface extension functions
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

/** \file
   Provides multiple hashing algorithms
   Interface extension functions - header file
*/

#ifndef FALCON_FEATHERS_HASH_FM_H
#define FALCON_FEATHERS_HASH_FM_H

#include <falcon/module.h>
#include <falcon/class.h>

namespace Falcon {
namespace Feathers {

class ModuleHash: public Module
{
public:
   Class* m_baseHashCls;

   ModuleHash();
   ~ModuleHash();
};

}
}

#endif

/* end of hash_fm.h */
