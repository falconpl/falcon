/*
   FALCON - The Falcon Programming Language.
   FILE: prototpye.h

   Prototype flexible class abstract type.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 06:35:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PROTOTYPE_H_
#define _FALCON_PROTOTYPE_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/flexyclass.h>
#include <falcon/pstep.h>


namespace Falcon
{

class FlexyDict;

/** Class holding polymorphic classes.
 A prototype is a set of 1 or more base classes, and just like an HyperClass it
 generates instances that actually hold the instances of the base classes.

 However, cotrarily to HyperClasses, Prototype classes can change their structure
 at runtime, and eventually propagate this change to child prototypes.

 Any change in the class structure is immediately reflected to all the children.

 \note Due to this dynamic nature, prototype strucutre access is interlocked via
 a per-class mutex.

  \note FlexClass (and Prototype) have bases, but not parents. getParent
 and isDerivedFrom won't be available on the classes, but only on proper
 instance and only at language level. However, getParentData is available
 (as it operates on a proper instance).
 */
class FALCON_DYN_CLASS PrototypeClass: public FlexyClass
{
public:
   /** Creates the prototype with a name and a master (final child) class.
 
    \note The master class is owned by this hyperclass and it's destroyed
    when the hyperclass is destroyed.
    */       
   PrototypeClass();
   virtual ~PrototypeClass();

   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   
   //=========================================================
   // Operators.
   //

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;

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
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;


   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_in( VMContext* ctx, void* self ) const;
   virtual void op_provides( VMContext* ctx, void* self, const String& property ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_iter( VMContext* ctx, void* self ) const;
   virtual void op_next( VMContext* ctx, void* self ) const;

private:

   class PStepGetPropertyNext: public PStep
   {
   public:
      PStepGetPropertyNext() { apply = apply_; }
      virtual ~PStepGetPropertyNext() {}
      void describeTo( String& text ) const { text = "PStepGetPropertyNext"; }
      static void apply_( const PStep* ps, VMContext* ctx );
   };

   PStepGetPropertyNext m_stepGetPropertyNext;

   inline bool callOverride( VMContext* ctx, FlexyDict* self, const String& opName, int count ) const;
   inline void override_unary( VMContext* ctx, void* instance, const String& opName ) const;
   inline void override_binary(  VMContext* ctx, void* instance, const String& opName ) const;
};

}

#endif /* _FALCON_PROTOTYPE_H_ */

/* end of prototype.h */
