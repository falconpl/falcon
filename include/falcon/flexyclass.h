/*
   FALCON - The Falcon Programming Language.
   FILE: flexyclass.h

   Class handling flexible objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 02:22:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FLEXYCLASS_H_
#define _FALCON_FLEXYCLASS_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon
{

/** Class handling flexible objects.

 \note FlexClass (and Prototype) have bases, but not parent. getParent
 and isDerivedFrom won't be available on the classes, but only on proper
 instance and only at language level. However, getParentData is available
 (as it operates on a proper instance).
 */
class FALCON_DYN_CLASS FlexyClass: public Class
{
public:

   FlexyClass();
   virtual ~FlexyClass();

   //====================================================================
   // Overrides from Class
   //
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;

   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual bool getProperty( const String& name, void* instance, Item& target ) const;
   virtual bool setProperty( const String& name, void* instance, const Item& target ) const;

   //=========================================================
   // Class management
   //

   virtual void gcMarkInstance( void* self, uint32 mark ) const;

   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void* getParentData( const Class* parent, void* data ) const;

   //=========================================================
   // Operators.
   //
   void op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool bOptional ) const;
   void delegate( void* instance, Item* target, const String& message ) const;

   virtual void op_neg( VMContext* ctx, void* self ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_mul( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;
   virtual void op_pow( VMContext* ctx, void* self ) const;
   virtual void op_shr( VMContext* ctx, void* self ) const;
   virtual void op_shl( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;
   virtual void op_apow( VMContext* ctx, void* self ) const;
   virtual void op_ashr( VMContext* ctx, void* self ) const;
   virtual void op_ashl( VMContext* ctx, void* self ) const;
   virtual void op_inc( VMContext* ctx, void* self ) const;
   virtual void op_dec( VMContext* ctx, void* self) const;
   virtual void op_incpost( VMContext* ctx, void* self ) const;
   virtual void op_decpost( VMContext* ctx, void* self ) const;
   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;
   // won't provide set/get property

   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_in( VMContext* ctx, void* self ) const;

   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_iter( VMContext* ctx, void* self ) const;
   virtual void op_next( VMContext* ctx, void* self ) const;
   
   // ==============================
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;

protected:
   FlexyClass( const String& name, int id = 0 );

private:
   inline bool operand( int opCount, const String& name, VMContext* ctx, void* self, bool bRaise = true ) const;
};

}

#endif /* _FALCON_FLEXYCLASS_H_ */

/* end of flexyclass.h */
