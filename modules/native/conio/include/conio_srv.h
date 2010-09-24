/*
   FALCON - The Falcon Programming Language.
   FILE: conio_srv.h

   Basic Console I/O support
   Interface extension functions
   -------------------------------------------------------------------
   Author: Unknown author
   Begin: Thu, 05 Sep 2008 20:12:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: The above AUTHOR

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

#ifndef conio_SRV_H
#define conio_SRV_H

#include <falcon/service.h>

namespace Falcon {
namespace Srv {

//forward for system specific data
class ConioSrvSys;

// provide a class that will serve as a service provider.
class ConsoleSrv: public Service
{
   class ConioSrvSys *m_sys;
   
public:
   typedef enum {
      e_none,
      e_init,
      e_not_init,
      e_dbl_init,
      e_read,
      e_write
   } error_type;
   
   // declare the name of the service as it will be published.
   ConsoleSrv();

   virtual ~ConsoleSrv();
   
   /** Initialize the console.
            On error during the initialization, returns e_init
         */
   virtual error_type init();
   
   /** Clears the screen. */
   virtual error_type cls();
   
   /** Turns off console services. */
   virtual void shutdown();   
};

}
}


#endif
/* end of conio_srv.h */
