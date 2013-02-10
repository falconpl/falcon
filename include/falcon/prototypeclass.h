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
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;

private:

   class PStepCallBaseInit: public PStep
   {
   public:
      PStepCallBaseInit() { apply = apply_; }
      virtual ~PStepCallBaseInit() {}
      void describeTo( String& text ) const { text = "PStepCallBaseInit"; }
      static void apply_( const PStep* ps, VMContext* ctx );
   };

   PStepCallBaseInit m_stepCallBaseInit;

   class PStepGetPropertyNext: public PStep
   {
   public:
      PStepGetPropertyNext() { apply = apply_; }
      virtual ~PStepGetPropertyNext() {}
      void describeTo( String& text ) const { text = "PStepGetPropertyNext"; }
      static void apply_( const PStep* ps, VMContext* ctx );
   };

   PStepGetPropertyNext m_stepGetPropertyNext;
};

}

#endif /* _FALCON_PROTOTYPE_H_ */

/* end of prototype.h */
