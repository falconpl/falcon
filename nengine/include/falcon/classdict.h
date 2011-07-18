/*
   FALCON - The Falcon Programming Language.
   FILE: classdict.h

   Standard language dictionary object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 14:15:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSDICT_H_
#define _FALCON_CLASSDICT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a dictionary as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassDict: public Class
{
public:

   ClassDict();
   virtual ~ClassDict();

   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   //virtual int compare( void* self, const Item& value ) const;

   //=============================================================

   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;

private:
};

}

#endif /* _FALCON_CLASSDICT_H_ */

/* end of classdict.h */
