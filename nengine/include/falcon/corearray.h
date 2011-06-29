/*
   FALCON - The Falcon Programming Language.
   FILE: corearray.h

   Standard language array object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 14:15:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREARRAY_H_
#define _FALCON_COREARRAY_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

class ItemArray;

/**
 Class handling an array as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreArray: public Class
{
public:

   CoreArray();
   virtual ~CoreArray();
   
   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;

   //=============================================================

   virtual void op_create( VMachine *vm, int32 pcount ) const;
   virtual void op_add( VMachine *vm, void* self ) const;
   virtual void op_aadd( VMachine *vm, void* self ) const;
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;

   virtual void op_getProperty( VMachine *vm, void* self, const String& prop) const;
   virtual void op_getIndex( VMachine *vm, void* self ) const;
   virtual void op_setIndex( VMachine *vm, void* self ) const;

private:
#if 0
   class FALCON_DYN_CLASS ToStringNextOp: public PStep {
   public:
      ToStringNextOp();
      static void apply_( const PStep*, VMachine* vm );
   } m_toStringNextOp;
#endif
   
};

}

#endif /* _FALCON_COREARRAY_H_ */

/* end of corearray.h */
