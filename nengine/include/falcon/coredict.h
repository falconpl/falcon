/*
   FALCON - The Falcon Programming Language.
   FILE: coredict.h

   Standard language dictionary object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 14:15:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREDICT_H_
#define _FALCON_COREDICT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a dictionary as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreDict: public Class
{
public:

   CoreDict();
   virtual ~CoreDict();

   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;

   //virtual int compare( void* self, const Item& value ) const;

   //=============================================================

   virtual void op_create( VMachine *vm, int32 pcount ) const;
   virtual void op_add( VMachine *vm, void* self ) const;
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;

   virtual void op_getProperty( VMachine *vm, void* self, const String& prop) const;
   virtual void op_getIndex(VMachine *vm, void* self ) const;
   virtual void op_setIndex(VMachine *vm, void* self ) const;

private:
};

}

#endif /* _FALCON_COREDICT_H_ */

/* end of coredict.h */
