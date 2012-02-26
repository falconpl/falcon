/*
   FALCON - The Falcon Programming Language.
   FILE: classarray.h

   Standard language array object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 14:15:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSARRAY_H_
#define _FALCON_CLASSARRAY_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

class ItemArray;

/**
 Class handling an array as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassArray: public Class
{
public:

   ClassArray();
   virtual ~ClassArray();
   
   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;

   //=============================================================

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;
   virtual void op_eval( VMContext* ctx, void* self ) const;

   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;

   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;

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

#endif /* _FALCON_CLASSARRAY_H_ */

/* end of classarray.h */
