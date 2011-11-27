/*
   FALCON - The Falcon Programming Language.
   FILE: inheritance.h

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 13:35:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_INHERITANCE_H_
#define _FALCON_INHERITANCE_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/pstep.h>
#include <falcon/sourceref.h>
#include <falcon/requirement.h>

namespace Falcon
{

class Class;
class Expression;
class PCode;

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
class FALCON_DYN_CLASS Inheritance
{
public:
   /** Creates the inheritance instance.
    \param name The logical name of the parent class.
    \param parent The class corresponding to the inheritance name, if known.
    \param owner The class where this inheritance is delcaraed.

    The inheritance name is the name of the class as it's written
    after the "from" clause in the inheritance declarartion.

    It may include namespaces and/or remote module names, and might
    not corespond with the name of the parent class as it is declared
    in the source module.

    */
   Inheritance( const String& name, Class* parent=0, Class* owner=0 );
   ~Inheritance();

   /** The name of the class that we're searching.
    This includes full path/namespace.

    The inheritance name is the name of the class as it's written
    after the "from" clause in the inheritance declarartion.

    It may include namespaces and/or remote module names, and might
    not corespond with the name of the parent class as it is declared
    in the source module.
    */
   const String& className() const { return m_name; }

   /** The parent class.
    \return the Parent class, when resolved, or 0 if still not available.
    */
   Class* parent() const { return m_parent; }

   /** Sets the parent actually reference by this inheritance.
    \param cls The class that the owner class derivates from.
    */
   void parent( Class* cls );

   /** Adds a parameter declaration.
      \param expr The expression that must be evaluated to generate the paramter.

    Inheritance parameters are expressions that must be calculated at runtime
    before invoking the init method of the subclass.
    */
   void addParameter( Expression* expr );

   /** Returns the number of parameters required to construct this inheritance.
    */
   size_t paramCount() const;
   
   /** Return the nth parameter required to construct this inheritance.
      \note The method will crash if n is out of range.
    */
   Expression* param( size_t n ) const;


   /** Gets the PCode generating the expression call stack.
    \return a PCode filled with the expressions building the initializatio parameters.

    The returned pcode is filled (or will be filled) with the pre-compilation
    of the expressions that shall be used as parameters for the call of the
    class constructor.

    It can be directly used in a syntree (or in a statement) to generate all
    the parameters in the final call stack.

    \note The returned entity is property of this class.
    */
   PCode* compiledExpr() const;

   /** Describes this inheritance.
      \param target A string where to place the description of this class.
    */
   void describe( String& target ) const;

   /** Describes this inheritance entry.
    \return A description of this entry.
    */
   String describe() const { 
      String target;
      describe( target );
      return target;
   }

   /** Sets the owner of this inheritance.
    \param cls The class resolving this inheritance.
    \see FalconClass::onInheritanceResolved
    */
   void owner( Class* cls ) { m_owner = cls; }

   /** Returns the owner of this inheritance.
    */
   Class* owner() const { return m_owner; }

   /** Add source line definition for this Inheritance. */
   void defineAt( int32 line, int32 chr ) { m_sdef.line( line ); m_sdef.chr(chr); }
   
   /** sourceRef. */
   const SourceRef& sourceRef() const { return m_sdef; }
   
   /** Return the requirement relative to this inheritance.
    
    This returns a functor that gets called back when a pending inheritance
    is resolved across modules.
    Used during the link process.
    */
   const Requirement& requirement() const { return m_requirer; }
   Requirement& requirement() { return m_requirer; }

private:

   class Private;
   Private* _p;

   String m_name;
   Class* m_parent;
   Class* m_owner;
   SourceRef m_sdef;
   
   class IRequirement: public Requirement
   {
   public:
      IRequirement( const String& name, Inheritance* owner ): 
         Requirement( name ),
         m_owner( owner ) 
      {}      
      virtual ~IRequirement() {}
      
      virtual void onResolved( const Module* source, const Symbol* srcSym, Module* tgt, Symbol* extSym );
   
   private:
      Inheritance* m_owner;
   }
   m_requirer;
   friend class IRequirement;
};

}

#endif /* _FALCON_INHERITANCE_H_ */

/* end of inheritance.h */
