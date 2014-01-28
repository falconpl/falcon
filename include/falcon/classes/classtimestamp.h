/* FALCON - The Falcon Programming Language.
 * FILE: classtimestamp.h
 * 
 * Interface extension functions
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Mon, 25 Mar 2013 00:47:03 +0100
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

#ifndef FALCON_CLASSTIMESTAMP_H
#define FALCON_CLASSTIMESTAMP_H

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/types.h>
#include <falcon/error.h>
#include <falcon/function.h>
#include <falcon/class.h>

namespace Falcon { 

class ClassTimeStamp: public Class
{
public:
   ClassTimeStamp();
   virtual ~ClassTimeStamp();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
};


} // namespace Falcon

#endif

/* end of classtimestamp.h */

