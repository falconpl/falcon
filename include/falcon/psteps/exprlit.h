/*
   FALCON - The Falcon Programming Language.
   FILE: exprlit.h

   Literal expression {(...)  expr } 
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
#include <falcon/varmap.h>
#include <falcon/psteps/exprunquote.h>

namespace Falcon {

class Symbol;

/** Literal expression.
 
 Can assume the following forms:
 \code
   {( par1, par2, ... ) expr }
   {( par1, par2, ... ) statement; statement ... }
   {[par1, par2, ...]  expr }
   {[par1, par2, ...]  statement; statement ... }
 \endcode
 */
class ExprLit: public Expression
{
public:
   ExprLit( int line=0, int chr=0 );
   ExprLit( TreeStep* expr, int line=0, int chr=0 );
   ExprLit( const ExprLit& other );
   
   virtual ~ExprLit();   
    
   virtual void describeTo( String&, int depth = 0 ) const;
    
   virtual Expression* clone() const { return new ExprLit(*this); }
   inline virtual bool isStandAlone() const { return true; }
   virtual bool isStatic() const {return false; }
   virtual bool simplify( Item& ) const { return false; }      
      
   /** This is actually a proxy to first() used during deserialization. */
   void setChild( TreeStep* st );   
      
   /**
    Adds a parameter to this parametric  expression.
    */
   void addParam( const String& name ) { m_paramTable.addParam(name); }
   
   /**
    Retrns the count of parameters.
    */
   int paramCount() const { return m_paramTable.paramCount(); }
   
   /**
    Gets the nth parameter.
    */
   const String& param( int n ) const { return m_paramTable.getParamName(n);}
   
   /**
    Declares an unquoted expression in the scope of this literal.
    */
   void registerUnquote( ExprUnquote* expr );
   
   /** Return the child attached to this literal.    
    */
   TreeStep* child() const { return m_child; }

   /**
    Returns true if the expression is eta.    
    Eta expressions pass untranslated parameters to the evaluation.
    */
   bool isEta() const { return m_paramTable.isEta(); }
   
   void setEta( bool e ) { m_paramTable.setEta(e); }

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );

private:
   TreeStep* m_child;
   VarMap m_paramTable;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif	/* _FALCON_EXPRLIT_H_ */

/* end of exprlit.h */
