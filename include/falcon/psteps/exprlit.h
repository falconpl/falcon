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

namespace Falcon {

/** Literal expression.
 
 Can assume the following forms:
 \code
   {( par1, par2, ... ) expr }
   {( par1, par2, ... ) statement; statement ... }
   {[par1, par2, ...]  expr }
   {[par1, par2, ...]  statement; statement ... }
 \endcode

 Soruce declaration of epex is also a literal expression:
 \code
   ^(p0,p1...)
 \endcode

 The former expression generates an ExprTree, the latter an ExprEP.

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

   void registerUnquote( Expression* unquoted );
   uint32 unquotedCount();
   Expression* unquoted( uint32 i );

   void setChild( TreeStep* st );
   TreeStep* child() const { return m_child; }

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );

private:
   class Private;
   ExprLit::Private* _p;
   TreeStep* m_child;

   void searchUnquotes( TreeStep* child );

   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif	/* _FALCON_EXPRLIT_H_ */

/* end of exprlit.h */
