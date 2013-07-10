/*
   FALCON - The Falcon Programming Language.
   FILE: bitbuf_ext.h

   Buffering extensions
   Bit-perfect buffer class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Jul 2013 13:22:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

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
   Buffering extensions
   Interface extension functions - header file
*/

#ifndef bufext_ext_H
#define bufext_ext_H

#include <falcon/types.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/classes/classerror.h>
#include <falcon/class.h>

namespace Falcon {
namespace Ext {

class ClassBitBuf: public ::Falcon::Class
{
public:
   ClassBitBuf();
   virtual ~ClassBitBuf();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};

Class* init_classbitbuf();

}} // namespace Falcon::Ext

#endif

/* end of bufext_ext.h */
