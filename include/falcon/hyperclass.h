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
#include <falcon/classes/classmulti.h>
#include <falcon/pstep.h>

namespace Falcon
{

class Function;
class DataReader;
class DataWriter;
class VMContext;
class FalconClass;
class Expression;
class ExprParentship;
class MetaHyperClass;

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
class FALCON_DYN_CLASS HyperClass: public ClassMulti
{
public:

   /** Creates a stand-alone hyperclass.
    \param name The name under which this class is known (can be the same as
           the name of the master class.

    This version of the constructor can be used to synthezise a hyperclass
    from code obviating the need to create a FalconClass prior to it. It is
    then possible to directly add parents to this hyperclass. This helps to
    create derived class from multiple parents in third party modules.
    */
   HyperClass( const String& name );

   virtual ~HyperClass();

   Function* constructor() const { return m_constructor; }

   /** Sets a topmost constructor function.
    This method can be used to set an initialization function that is
    called above the master class constructor, in case the hyperclass
    is synthezized.
    */
   void constructor( Function* c ) { m_constructor = c; }

   bool addParent( Class* parent );

   bool addProperty( const String& name, const Item& initValue );
   bool addProperty( const String& name, Expression* initExpr );
   bool addProperty( const String& name );
   bool addMethod( Function* mth );

   //=========================================
   // Instance management
   virtual Class* getParent( const String& ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;


   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual void* getParentData( const Class* parent, void* data ) const;

   //=========================================================
   // Class management
   //

   virtual void gcMark( uint32 mark );
   virtual void gcMarkInstance( void* self, uint32 mark ) const;

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

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;

protected:
   void addParentProperties( Class* cls );
   void addParentProperty( Class* cls, const String& pname );
   Class* getParentAt( int pos ) const;

private:
   Function* m_constructor;
   ExprParentship* m_parentship;
   FalconClass* m_master;
   int m_nParents;
   bool m_ownParentship;

   /** Creates the hyperclass with a name and a master (final child) class.
    \param master The master class.

    The master class is owned by this hyperclass and it's destroyed
    when the hyperclass is destroyed.
    */
   HyperClass( FalconClass* master );

   void setParentship( ExprParentship* ps, bool own = true );

   class FALCON_DYN_CLASS InitParentsStep: public PStep
   {
   public:
      InitParentsStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~InitParentsStep() {};
      virtual void describeTo( String& tgt ) const {
         tgt = "InitParentStep for " + m_owner->name();
      }

      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   InitParentsStep m_initParentsStep;

   class FALCON_DYN_CLASS InitMasterExprStep: public PStep
   {
   public:
      InitMasterExprStep( HyperClass* o ): m_owner(o) { apply = apply_; }
      virtual ~InitMasterExprStep() {};
      virtual void describeTo( String& tgt ) const {
         tgt = "InitMasterExprStep for " + m_owner->name();
      }

      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      HyperClass* m_owner;
   };

   InitMasterExprStep m_InitMasterExprStep;

   friend class FalconClass;
   friend class MetaHyperClass;
};

}

#endif /* _FALCON_HYPERCLASS_H_ */

/* end of hyperclass.h */
