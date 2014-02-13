/*
   FALCON - The Falcon Programming Language.
   FILE: exprsym.h

   Syntactic tree item definitions -- expression elements -- symbol.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRSYM_H
#define FALCON_EXPRSYM_H

#include <falcon/setup.h>
#include <falcon/expression.h>
#include <falcon/gclock.h>

namespace Falcon {

class Symbol;

/** Expression referring to a symbol.
 
 These expressions have the role to access a symbol in read
 or write (lvalue) mode in an expression.
 
 Initially, when first created in an expression, the ExprSymbol
 MAY refer just the name of the symbol that should be referenced,
 so that the real symbol can be resolved at a later stage.
 
 The expressions will either hold a symbol coming from a container
 (a function or a module), or a dynamic symbol created from the script.
 
 In the former case, the symbol is simply referenced directly, without
 any marking, GC Locking or reference counting mechanism. This is because
 any element of a tree will end up in marking its container, where the
 symbol referenced by this expression is supposed to be. Also, trees cannot
 be unparented. This is achieved through the referenceFromContainer method.
 
 In the latter case, the expression should create a GC Lock to keep safe
 the dynamically created symbol as long as the expression, and the tree it is
 in, exists. This is achieved through the safeGuard method.
 
 Whenever a tree is cloned (which is the only way to move a tree away from
 its container), if an ExprSymbol is holding a symbol, it creates a new
 dynamic symbol resembling the old one, it stores it in the garbage collector
 and gives it to the cloned ExprSymbol via the referenceFromContainer
 method.
 */
class FALCON_DYN_CLASS ExprSymbol: public Expression
{
public:
   ExprSymbol( int line = 0, int chr = 0 );
   
   /** Declare a forward symbol reference.
    \param name The name of the symbol that should be used in this expression.
    
    This constrcutor can be used to create a symbol placeholder in an expression,
    which can be then filled later on.
    */
   ExprSymbol( const String& name,  int line = 0, int chr = 0 );
   
   /** Declare A fully constructor symbol access expression.
    
    This constructor assigns the symbol directly, as if
      referenceFromContainer method was called.
    
    */
   ExprSymbol( const Symbol* target, int line = 0, int chr = 0 );
   ExprSymbol( const ExprSymbol& other );
   virtual ~ExprSymbol();

   virtual void render( TextWriter* tw, int32 depth ) const;

   inline virtual ExprSymbol* clone() const { return new ExprSymbol(*this); }

   /** Symbols cannot be simplified. */
   inline virtual bool simplify( Item& ) const { return false; }
   virtual bool fullDefining() { return true; }

   // Return the symbol pointed by this expression.
   const Symbol* symbol() const { return m_symbol; }
   void symbol( const Symbol* sym );
   
   /** Returns the symbol name associated with this expression.
    \return A symbol name.
    
    If a symbol has been given for this expression, then the name of
    that symbol is returned, otherwise the name assigned by the constructor
    is returned.
    .*/
   const String& name() const;
   
   void name( const String& newName );

   bool isPure() const {return m_pure;}
   void setPure( bool m ) { m_pure = m; }

protected:
   const Symbol* m_symbol;
   bool m_pure;
   
   static void apply_( const PStep* ps, VMContext* ctx );
   
   class FALCON_DYN_CLASS PStepLValue: public PStep
   {
   public:
      ExprSymbol* m_owner;
      
      PStepLValue( ExprSymbol* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepLValue(){}
      virtual void describeTo( String& ) const;
      static void apply_( const PStep* ps, VMContext* ctx );
   };   
   
   PStepLValue m_pslv;
};

}

#endif

/* end of exprsym.h */
