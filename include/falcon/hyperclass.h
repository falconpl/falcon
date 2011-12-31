/*
   FALCON - The Falcon Programming Language.
   FILE: hyperclass.h

   Class holding more user-type classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Jul 2011 11:56:21 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_HYPERCLASS_H_
#define _FALCON_HYPERCLASS_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/multiclass.h>
#include <falcon/pstep.h>

namespace Falcon
{

class Function;
class DataReader;
class DataWriter;
class VMContext;
class Inheritance;


/** Class holding more user-type classes.

 Hyperclasses are a common skeleton for classes formed by the union of
 exactly one child (the master class) and all its parents. At engine level,
 the HyperClass appare like the "child" class empowered with the methods and
 properties from the parent.

 As the nature of the child and the parents is transparent to a hyperclass,
 classes of any kind can be used to create an hyperclass. However, as the
 hyperclass must know in advance the properties offered by all the classes,
 subclasses that provide a varying set of properties are not served correctly
 by hyperclasses. Those varying classes aren't usually suited to form a rigid
 hierarchy as the one represented by HyperClass, and are usually left alone
 (as they already provide polimorphic abilities and can be shaped on the
 need of the moment, obviating the need of inheritance).

 \note Hyperclasses don't offer any mechanism to support forward definition
 of parents. The final child class and the parent classes must be known
 by the time the hyperclass is formed.

 The instance created by the hyperclass is actually a plain item array, where
 each element is the "self" seen by each class. Altough a bit sub-optimal
 in some cases, this helps to transparently hold Falcon item types as subclasses,
 and simplifies garbage collecting management in case some subclass requires
 the generated self instance not to be marked.
 
 */
class FALCON_DYN_CLASS HyperClass: public MultiClass
{
public:
   /** Creates the hyperclass with a name and a master (final child) class.
    \param name The name under which this class is known (can be the same as
           the name of the master class.
    \param master The master class.

    \note The master class is owned by this hyperclass and it's destroyed
    when the hyperclass is destroyed.
    */
   HyperClass( const String& name, Class* master );
   virtual ~HyperClass();

   /** Adds a parent and the parentship declaration.
    \param cls The class.
    \return true If the name of the remote class is free, false if it was
                 already assigned.

    As the parent is created, all the properties that were not declared elsewhere
    are imported, as well as a property holding the same name of the parent
    class that resolves to the class itself.

    \note The priority of the properties is first-to-last. This means you
          must add as parents classes with higher priority first.
    */
   bool addParent( Inheritance* cls );

   //=========================================
   // Instance management
   virtual Class* getParent( const String& ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual void* getParentData( Class* parent, void* data ) const;

   Function* constructor() const { return m_constructor; }
   void constructor( Function* c ) { m_constructor = c; }
   
   //=========================================================
   // Class management
   //

   virtual void gcMarkMyself( uint32 mark );

   virtual void gcMark( void* self, uint32 mark ) const;

   /** List all the properties in this class.
     @param self An instance (actually, it's unused as the class knows its properties).
     @param cb A callback function receiving one property at a time.
    */
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;

   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   //=========================================================
   // Operators.
   //

   virtual void op_create( VMContext* ctx, int32 pcount ) const;

protected:
   
   void addParentProperties( Class* cls );
   void addParentProperty( Class* cls, const String& pname );
   Class* getParentAt( int pos ) const;

private:

   class Private;
   friend class Private;
   Private* _p;

   Class* m_master;
   int m_nParents;
   Function* m_constructor;   

   class FALCON_DYN_CLASS FinishCreateStep: public PStep
   {
   public:
      FinishCreateStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~FinishCreateStep() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   class FALCON_DYN_CLASS CreateMasterStep: public PStep
   {
   public:
      CreateMasterStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~CreateMasterStep() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   class FALCON_DYN_CLASS ParentCreatedStep: public PStep
   {
   public:
      ParentCreatedStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~ParentCreatedStep() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   class FALCON_DYN_CLASS CreateParentStep: public PStep
   {
   public:
      CreateParentStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~CreateParentStep() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   class FALCON_DYN_CLASS FinishInvokeStep: public PStep
   {
   public:
      FinishInvokeStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~FinishInvokeStep() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   class FALCON_DYN_CLASS InvokeMasterStep: public PStep
   {
   public:
      InvokeMasterStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~InvokeMasterStep() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   class FALCON_DYN_CLASS CreateEmptyNext: public PStep
   {
   public:
      CreateEmptyNext( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~CreateEmptyNext() {};
      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   FinishCreateStep m_finishCreateStep;
   CreateMasterStep m_createMasterStep;
   ParentCreatedStep m_parentCreatedStep;
   CreateParentStep m_createParentStep;

   FinishInvokeStep m_finishInvokeStep;
   InvokeMasterStep m_invokeMasterStep;
   CreateEmptyNext m_createEmptyNext;

   friend class FinishCreateStep;
   friend class CreateMasterStep;
   friend class ParentCreatedStep;
   friend class CreateParentStep;

   friend class FinishInvokeStep;
   friend class InvokeMasterStep;
   friend class CreateEmptyNext;
};

}

#endif /* _FALCON_HYPERCLASS_H_ */

/* end of hyperclass.h */
