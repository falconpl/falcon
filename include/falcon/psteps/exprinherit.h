/*
   FALCON - The Falcon Programming Language.
   FILE: exprinherit.h

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 13:35:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRINHERIT_H_
#define _FALCON_EXPRINHERIT_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/pstep.h>
#include <falcon/sourceref.h>
#include <falcon/requirement.h>
#include <falcon/psteps/exprvector.h>

namespace Falcon
{

class Class;
class Expression;
class VMContext;
class ItemArray;

/** Structure holding information about inheritance in a class.
 This structure holds the needed information to create automatic inheritance
 structures. It holds possibly forward reference to the base class (parent)
 and it owns the expressions that are needed to invoke their constructor.

 This class is mainly used in the FalconClass to allow delayed resolutiuon of
 parent classes, but it is also known by the HyperClass structure and can
 be used by any user class requiring to have knowledge about "base classes",
 and instruction on how to instantiate them.

 \note The inheritance doesn't own the classes it refers to, but it owns
 the expressions used to automatically invoke the base class constructors
 ('parameters').
 
 */
class FALCON_DYN_CLASS ExprInherit: public ExprVector
{
public:
   ExprInherit( int line=0, int chr=0 );
   ExprInherit( const String& name, int line=0, int chr=0 );
   ExprInherit( Class* base, int line=0, int chr=0 );
   ExprInherit( const ExprInherit& other );
   
   virtual ~ExprInherit();

   const String& name() const { return m_name; }
   
   /** The parent class.
    \return the Parent class, when resolved, or 0 if still not available.
    */
   Class* base() const { return m_base; }

   /** Sets the parent actually reference by this inheritance.
    \param cls The class that the owner class derivates from.
    */
   void base( Class* cls );

   virtual void describeTo( String& target, int depth = 0 ) const;

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& ) const { return false; }  
   virtual ExprInherit* clone() const { return new ExprInherit(*this); }
   
   /** Creats a dynamic requirement for a missing base class in this expression.
    This equates to a forward declaration of the base class.
    */
   Requirement* makeRequirement( Class* target );
   
   /** Specify if this inheritance was involved in a forward definition.
    Elements that were involved in a requirement shall not serialize
    their internal data as they were resolved, but as they were originally
    created.
    */
   bool hadRequirement() const { return m_bHadRequirement; }
   
   /** Sets the requirement status of this inheritance.
    Used during de-serialization to restore the status of this inheritance
    in the host module.
    */
   void hadRequirement( bool b ) { m_bHadRequirement = b; }
   
   class FALCON_DYN_CLASS IRequirement: public Requirement
   {
   public:
      IRequirement( const String& name ):
         Requirement( name ),
            m_owner(0),
            m_target(0)         
         {}
      
      IRequirement( ExprInherit* owner, Class* target ): 
         Requirement( owner->name() ),
         m_owner( owner ),
         m_target( target )
      {}      
      virtual ~IRequirement() {}
      
      virtual void onResolved( const Module* sourceModule, const String& sourceName, Module* targetModule, const Item& value, const Variable* targetVar );
      virtual Class* cls() const;
      static void registerMantra();
   private:
      
      ExprInherit* m_owner;
      Class* m_target;
      class ClassIRequirement;
      friend class ClassIRequirement;
      static Class* m_mantraClass;
   };
   
private:
   Class* m_base;
   String m_name;
   bool m_bHadRequirement;
   
   
   
   friend class IRequirement;   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif /* _FALCON_INHERITANCE_H_ */

/* end of exprinherit.h */
