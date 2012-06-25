/*
   FALCON - The Falcon Programming Language.
   FILE: exprlit.h

   Evaluation expression (^* expr) 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 04 Jan 2012 00:55:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRLIT_H_
#define _FALCON_EXPRLIT_H_

#include <falcon/setup.h>
#include <falcon/expression.h>
#include <falcon/symboltable.h>

namespace Falcon {

class Symbol;

/** Literal expression.
 
 Can assume the following forms:
 \code
   {~ expr }
   {~ statement; statement ... }
   {~~ par1, par2, ... => expr }
   {~~ par1, par2, ... => statement; statement ... }
   {~~ par1, par2, ... >> expr }
   {~~ par1, par2, ... >> statement; statement ... }
 
 */
class ExprLit: public Expression
{
public:
   ExprLit( int line=0, int chr=0 );
   ExprLit( TreeStep* expr, int line=0, int chr=0 );
   ExprLit( const ExprLit& other );
   
   virtual ~ExprLit() {};   
    
   virtual void describeTo( String&, int depth = 0 ) const;
    
   virtual Expression* clone() const { return new ExprLit(*this); }
   inline virtual bool isStandAlone() const { return true; }
   virtual bool isStatic() const {return false; }
   virtual bool simplify( Item& ) const { return false; }      
      
   /** This is actually a proxy to first() used during deserialization. */
   void setChild( TreeStep* st );   
   
   /** Creates a new dynamic symbol, or returns a previously created one.    
    */
   Symbol* makeSymbol( const String& name, int declLine );
   
   /**
    Adds a parameter to this parametric  expression.
    */
   void addParam( const String& name );
   
   /**
    Retrns the count of parameters.
    */
   int paramCount() const { return m_paramTable.localCount(); }
   
   /**
    Gets the nth parameter.
    */
   Symbol* param( int n ) const { return m_paramTable.getLocal(n);}
   
   /**
    Declares an unquoted expression in the scope of this literal.
    */
   virtual void subscribeUnquote( Expression* expr );
   
   /** Return the child attached to this literal.
    
    */
   TreeStep* child() const { return m_child; }

   /**
    Returns true if the expression is eta.    
    Eta expressions pass untranslated parameters to the evaluation.
    */
   bool isEta() const { return m_isEta; }
   
   void setEta( bool e ) { m_isEta = e; }
   
public:
   class Private;
   ExprLit::Private* _p;
  
   TreeStep* m_child;
   SymbolTable m_paramTable;
   bool m_isEta;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif	/* _FALCON_EXPRLIT_H_ */

/* end of exprlit.h */
