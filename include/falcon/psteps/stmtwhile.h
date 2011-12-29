/*
   FALCON - The Falcon Programming Language.
   FILE: stmtwhile.h

   Statatement -- while
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STATWHILE_H
#define FALCON_STATWHILE_H

#include <falcon/statement.h>

namespace Falcon
{

/** While statement.
 *
 * Loops in a set of statements (syntree) while the given expression evaluates as true.
 */
class FALCON_DYN_CLASS StmtWhile: public Statement
{
public:
   /** Creates a while statement, creating a new syntree. */   
   StmtWhile( Expression* expr, int32 line=0, int32 chr = 0 );
   
   /** Creates a while statement, adopting an existign statement set. */
   StmtWhile( Expression* expr, SynTree* stmts, int32 line=0, int32 chr = 0 );
   virtual ~StmtWhile();

   void describeTo( String& tgt, int depth=0 ) const;
   void oneLinerTo( String& tgt ) const;
   static void apply_( const PStep*, VMContext* ctx );

   SynTree* mainBlock() const { return m_stmts; }
   
   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool nth( int32 n, TreeStep* ts );
   
   
   virtual int32 arity() const;
   
   /** Nth sub-element of this element in 0..arity() */
   virtual TreeStep* nth( int32 n ) const;
   
   /** Setting the nth sub-element.
    \param n The sub-element number.
    \param ts An unparented expression.
    \return true if \b ts can be parented and n is valid, false otherwise.
    
    If a previous expression occupies this position, it is destroyed.    
    */
   virtual bool nth( int32 n, TreeStep* ts );
   
   virtual Expression* selector();   
   virtual bool selector( Expression* e ); 
private:   
   SynTree* m_stmts;
   Expression* m_expr;
};

}

#endif

/* end of stmtwhile.h */
