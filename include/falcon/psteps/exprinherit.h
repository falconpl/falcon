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
#include <falcon/psteps/exprvector.h>

namespace Falcon
{

class Class;
class Expression;
class VMContext;
class ItemArray;
class Engine;

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
   ExprInherit( const Symbol* sym, int line=0, int chr=0 );
   ExprInherit( Class* base, int line=0, int chr=0 );
   ExprInherit( const ExprInherit& other );
   
   virtual ~ExprInherit();

   const Symbol* symbol() const { return m_symbol; }
   
   /** The parent class.
    \return the Parent class, when resolved, or 0 if still not available.
    */
   Class* base() const { return m_base; }

   /** Sets the parent actually reference by this inheritance.
    \param cls The class that the owner class derivates from.
    */
   void base( Class* cls );

   virtual void render( TextWriter* tw, int depth ) const;

   virtual bool simplify( Item& ) const { return false; }  
   virtual ExprInherit* clone() const { return new ExprInherit(*this); }

   
private:
   Class* m_base;
   const Symbol* m_symbol;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif /* _FALCON_INHERITANCE_H_ */

/* end of exprinherit.h */
