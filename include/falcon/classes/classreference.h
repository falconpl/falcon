/*
   FALCON - The Falcon Programming Language.
   FILE: classreference.h

   Reference to remote items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 10:53:54 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSREFERENCE_H_
#define _FALCON_CLASSREFERENCE_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/**
 Reference to remote items.
 */

class FALCON_DYN_CLASS ClassReference: public Class
{
public:

   ClassReference();
   virtual ~ClassReference();

   //=============================================================

   virtual Class* getParent( const String& name ) const;

   //=========================================
   // Instance management
   //

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual bool gcCheckInstance( void* self, uint32 mark ) const;
   
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
};

}

#endif /* _FALCON_REFERENCE_H_ */

/* end of classreference.h */
