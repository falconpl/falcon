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

namespace Falcon {

class Symbol;

/** Literal expression {~ expr } 
 */
class ExprLit: public UnaryExpression
{
public:
   ExprLit( int line=0, int chr=0 );
   ExprLit( Expression* expr, int line=0, int chr=0 );
   ExprLit( SynTree* st, int line=0, int chr=0 );
   ExprLit( const ExprLit& other );
   
   virtual ~ExprLit() {};   
    
   virtual void describeTo( String&, int depth = 0 ) const;
    
   virtual Expression* clone() const { return new ExprLit(*this); }
   inline virtual bool isStandAlone() const { return true; }
   virtual bool isStatic() const {return false; }
   virtual bool simplify( Item& ) const { return false; }      
   
   /** This is actually a proxy to first() used during deserialization. */
   void setExpression( Expression* expr );
   /** This is actually a proxy to first() used during deserialization. */
   void setTree( SynTree* st );

   /** Creates a new dynamic symbol, or returns a previously created one.
    
    */
   Symbol* makeSymbol( const String& name, int declLine );
   
   virtual void subscribeUnquote( Expression* expr );
public:
   class Private;
   ExprLit::Private* _p;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif	/* _FALCON_EXPRLIT_H_ */

/* end of exprlit.h */
