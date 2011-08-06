/*
   FALCON - The Falcon Programming Language.
   FILE: uri.h

   Falcon core module -- Interface to URI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_URI_H
#define FALCON_CORE_URI_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/class.h>

namespace Falcon {
namespace Ext {

class ClassURI: public Class
{
public:
   ClassURI();
   virtual ~ClassURI();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   
   virtual void gcMark( void* self, uint32 mark ) const;
   virtual bool gcCheck( void* self, uint32 mark ) const;
   
   //=============================================================
   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;   
};

}
}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of uri.h */
