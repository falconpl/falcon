/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Jul 2014 14:43:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: The above AUTHOR

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

#ifndef _FALCON_MODULES_CURL_FM_
#define _FALCON_MODULES_CURL_FM_

#include <falcon/module.h>
#include <falcon/error.h>

#ifndef FALCON_ERROR_CURL_BASE
#define FALCON_ERROR_CURL_BASE            2350
#endif

#define FALCON_ERROR_CURL_INIT            (FALCON_ERROR_CURL_BASE+0)
#define FALCON_ERROR_CURL_EXEC            (FALCON_ERROR_CURL_BASE+1)
#define FALCON_ERROR_CURL_PM              (FALCON_ERROR_CURL_BASE+2)
#define FALCON_ERROR_CURL_SETOPT          (FALCON_ERROR_CURL_BASE+3)
#define FALCON_ERROR_CURL_GETINFO         (FALCON_ERROR_CURL_BASE+4)
#define FALCON_ERROR_CURL_HISIN           (FALCON_ERROR_CURL_BASE+5)
#define FALCON_ERROR_CURL_HNOIN           (FALCON_ERROR_CURL_BASE+6)
#define FALCON_ERROR_CURL_MULTI           (FALCON_ERROR_CURL_BASE+7)

namespace Falcon {
namespace Canonical {

class ModuleCurl: public Falcon::Module
{
public:
   ModuleCurl();
   virtual ~ModuleCurl();

   Class* handleClass() const { return m_handleClass; }
   Class* multiClass() const { return m_multiClass; }

private:
   static int init_count;

   Class* m_handleClass;
   Class* m_multiClass;
};

}
}

#endif
