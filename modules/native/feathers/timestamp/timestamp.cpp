/* FALCON - The Falcon Programming Language.
 * FILE: timestamp.cpp
 * 
 * Gregorian calendar support
 * Main module file, providing the module object to the Falcon engine.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 24 Mar 2013 18:57:16 +0100
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

#include <falcon/module.h>
#include "timestamp_ext.h"

#include "version.h"


//Define the math_extra module class
class TimeStampModule: public Falcon::Module
{
public:
   // initialize the module
   TimeStampModule():
      Module("timestamp")
   {
      // Standard

      *this
         << new Falcon::Ext::FALCON_FUNCTION_NAME(currentTime)
         << new Falcon::Ext::FALCON_FUNCTION_NAME(parseRFC2822)

         << new Falcon::Ext::ClassTimeStamp
         << new Falcon::Ext::ClassTimeZone
               ;

   }

   virtual ~TimeStampModule() {}
};

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new TimeStampModule;
   return mod;
}

/* end of timestamp.cpp */

