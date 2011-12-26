/*
   FALCON - The Falcon Programming Language.
   FILE: exprref.h

   Syntactic tree item definitions -- reference to symbols.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 11:51:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRREF_H
#define FALCON_EXPRREF_H

#include <falcon/psteps/exprsym.h>

namespace Falcon
{

class Symbol;

/** Expression declaring arrays or list of expressions.

 This class is used both to store array declarations and list of expressions,
 for example when generating parametric calls or assignment lists.

 */
class FALCON_DYN_CLASS ExprRef: public Expression
{
public:
   ExprRef( Symbol* sym );
   ExprRef( ExprSymbol* sym );
   ExprRef( const ExprRef& other );
   
   virtual ~ExprRef();
   
   static void apply_( const PStep*, VMContext* ctx );

   inline virtual ExprRef* clone() const { return new ExprRef( *this ); }
   
   virtual void describeTo( String&, int depth ) const;

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;

   inline Symbol* symbol() const { return m_symbol; }

private:
   Symbol* m_symbol;
   ExprSymbol* m_expr;
};

}

#endif

/* end of exprref.h */
