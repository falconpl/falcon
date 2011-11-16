/*
   FALCON - The Falcon Programming Language.
   FILE: exprsym.h

   Syntactic tree item definitions -- expression elements -- symbol.

   Pure virtual class base for the various symbol types.
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

namespace Falcon {

class Symbol;
class DynSymbol;
class GlobalSymbol;
class LocalSymbol;
class ClosedSymbol;


/** Expression referring to a symbol.
 
 These expressions have the role to access a symbol in read
 or write (lvalue) mode in an expression.
 
 Initially, when first created in an expression, the ExprSymbol
 MAY refer just the name of the symbol that should be referenced,
 so that the real symbol can be resolved at a later stage.
 
 \todo Optimize adding PSTeps to access directly local, global and closed symbol
 values. Now we're using the symbol access value functions, but that's superfluous
 in a release optimized POV.
 
 \note Expressions NEVER own the symbols they refer to. They must be held
 somewhere else (in the compiler context, module or interactive virtual
 machine).
 */
class FALCON_DYN_CLASS ExprSymbol: public Expression
{
public:
   /** Declare a forward symbol reference.
    \param name The name of the symbol that should be used in this expression.
    
    This constrcutor can be used to create a symbol placeholder in an expression,
    which can be then filled later on.
    */
   ExprSymbol( const String& name );
   /** Declare A fully constructer symbol access expression.
    */
   ExprSymbol( Symbol* target );
   ExprSymbol( const ExprSymbol& other );
   virtual ~ExprSymbol();

   virtual void describeTo(String & str) const;

   inline virtual ExprSymbol* clone() const { return new ExprSymbol(*this); }

   /** Symbols cannot be simplified. */
   inline virtual bool simplify( Item& ) const { return false; }
   inline virtual bool isStatic() const { return false; }

   virtual void serialize( DataWriter* s ) const;

   // Return the symbol pointed by this expression.
   Symbol* symbol() const { return m_symbol; }
   void symbol( Symbol* sym) { m_symbol = sym; }
   
   /** Returns the symbol name associated with this expression.
    \return A symbol name.
    
    If a symbol has been given for this expression, then the name of
    that symbol is returned, otherwise the name assigned by the constructor
    is returned.
    .*/
   const String& name() const;

   void precompileLvalue( PCode* pcode ) const;

   virtual void precompileAutoLvalue( PCode* pcode, const PStep* activity, bool bIsBinary, bool bSaveOld ) const;

protected:
   String m_name;
   Symbol* m_symbol;
   
   static void apply_( const PStep* ps, VMContext* ctx );
   
   class FALCON_DYN_CLASS PStepLValue: public PStep
   {
   public:
      ExprSymbol* m_owner;
      
      PStepLValue( ExprSymbol* owner ): m_owner(owner) { apply = apply_; }
      virtual void describeTo( String& ) const;
      static void apply_( const PStep* ps, VMContext* ctx );
   };   
   
   class FALCON_DYN_CLASS PStepSave: public PStep
   {
   public:
      PStepSave(){ apply = apply_; }
      virtual void describeTo( String& ) const;
      static void apply_( const PStep* ps, VMContext* ctx );
   };  
   
   class FALCON_DYN_CLASS PStepRemove: public PStep
   {
   public:
      PStepRemove() { apply = apply_; }
      virtual void describeTo( String& ) const;
      static void apply_( const PStep* ps, VMContext* ctx );
   };  
   
   PStepLValue m_pslv;
   PStepSave m_pstepSave;        
   PStepRemove m_pstepRemove;
        
   virtual void deserialize( DataReader* s );
   inline ExprSymbol( operator_t type ):
      Expression( type ),
      m_pslv(this)
   {}

};

}

#endif

/* end of exprsym.h */
