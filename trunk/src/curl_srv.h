/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 27 Nov 2009 16:31:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

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
   Service publishing - reuse Falcon module logic (mod) in
   your applications!
*/

#ifndef curl_SRV_H
#define curl_SRV_H

#include <falcon/service.h>

namespace Falcon {
namespace Srv {

// provide a class that will serve as a service provider.
class Skeleton: public Service
{
public:

   // declare the name of the service as it will be published.
   Skeleton():
      Service( "Skeleton" )
   {}

   // Provide here methods that needs to be exported.
   int skeleton();
};

}
}


#endif
/* end of curl_srv.h */
