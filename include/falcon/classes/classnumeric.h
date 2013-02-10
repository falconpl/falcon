/*
 FALCON - The Falcon Programming Language.
 FILE: classnumeric.h
 
 Int object handler.
 -------------------------------------------------------------------
 Author: Francesco Magliocca
 Begin: Sat, 11 Jun 2011 21:45:01 +0200
 
 -------------------------------------------------------------------
 (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#ifndef _FALCON_CLASSNUMERIC_H_
#define _FALCON_CLASSNUMERIC_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{
    
/**
Class handling a numeric as an item in a falcon script.
*/

class FALCON_DYN_CLASS ClassNumeric: public Class
{
public:

   ClassNumeric();
   virtual ~ClassNumeric();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext*, DataWriter* dw, void* data ) const;
   virtual void restore( VMContext* , DataReader* dr) const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   virtual Class* getParent( const String& name ) const;
   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual void enumerateParents( ClassEnumerator& cb ) const;
   virtual void* getParentData( Class* parent, void* data ) const;

   //=============================================================
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_mul( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;
   virtual void op_pow( VMContext* ctx, void* self ) const;
   virtual void op_shr( VMContext* ctx, void* self ) const;
   virtual void op_shl( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self ) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;
   virtual void op_apow( VMContext* ctx, void* self ) const;
   virtual void op_ashr( VMContext* ctx, void* self ) const;
   virtual void op_ashl( VMContext* ctx, void* self ) const;
   virtual void op_inc( VMContext* ctx, void* self ) const;
   virtual void op_dec( VMContext* ctx, void* self ) const;
   virtual void op_incpost( VMContext* ctx, void* self ) const;
   virtual void op_decpost( VMContext* ctx, void* self ) const;
};

}

#endif /* _FALCON_CLASSNUMERIC_H_ */

/* end of classnumeric.h */
