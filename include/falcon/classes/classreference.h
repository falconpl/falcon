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

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;  
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   // TODO: Flatten/unflatten.
   
   virtual void gcMark( void* self, uint32 mark ) const;
   virtual bool gcCheck( void* self, uint32 mark ) const;
   
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_neg( VMContext* ctx, void* self ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_mul( VMContext* ctx, void* self ) const;   
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;
   virtual void op_shr( VMContext* ctx, void* self ) const;
   virtual void op_shl( VMContext* ctx, void* self ) const;
   virtual void op_pow( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;
   virtual void op_apow( VMContext* ctx, void* self ) const;
   virtual void op_ashr( VMContext* ctx, void* self ) const;
   virtual void op_ashl( VMContext* ctx, void* self ) const;
   virtual void op_inc( VMContext* vm, void* self ) const;
   virtual void op_dec(VMContext* vm, void* self) const;
   virtual void op_incpost(VMContext* vm, void* self ) const;
   virtual void op_decpost(VMContext* vm, void* self ) const;
   virtual void op_getIndex(VMContext* vm, void* self ) const;
   virtual void op_setIndex(VMContext* vm, void* self ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;
   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_in( VMContext* ctx, void* self ) const;
   virtual void op_provides( VMContext* ctx, void* self, const String& property ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
   virtual void op_eval( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_iter( VMContext* ctx, void* self ) const;
   virtual void op_next( VMContext* ctx, void* self ) const;
   
};

}

#endif /* _FALCON_REFERENCE_H_ */

/* end of classreference.h */
