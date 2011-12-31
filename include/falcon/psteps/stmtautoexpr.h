/*
   FALCON - The Falcon Programming Language.
   FILE: stmtautoexpr.cpp

   Syntactic tree item definitions -- Autoexpression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STMTAUTOEXPR_H_
#define _FALCON_STMTAUTOEXPR_H_

#include <falcon/statement.h>
#include <falcon/expression.h>

namespace Falcon {
/** Autoexpression.
 *
 * This statement is needed to wrap an expression so that its
 * result is removed from the stack.
 *
 * The obvious un-necessity of this class advises for the usage of
 * registers where expressions place their results; but reasons
 * against are equally valid.
 */
class FALCON_DYN_CLASS StmtAutoexpr: public Statement
{
public:
   StmtAutoexpr( int32 line=0, int32 chr = 0 );
   StmtAutoexpr( Expression* expr, int32 line=0, int32 chr = 0 );
   StmtAutoexpr( const StmtAutoexpr& other );
   virtual ~StmtAutoexpr();

   virtual bool isAutoExpr() const { return true; }
   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;   

   /** Removes the expression stored in this AutoExpression.
    \return The held expression, or 0 if it was not set.

    This method can be used when the parser generated an autoexpression
    that is actually used elsewhere.
    */
   Expression* detachExpr() {
      Expression* expr = m_expr;
      m_expr->setParent(0);
      m_expr = 0;
      return expr;
   }
   
   virtual inline StmtAutoexpr* clone() const { return new StmtAutoexpr(*this); }
   
   void setInteractive( bool bInter );
   bool isInteractive() const { return m_bInteractive; }
   
   void setInRule( bool bRule );
   bool isInRule() const { return m_bInRule; }
   
   /** Returns the selector of this autoexpression. */
   virtual Expression* selector() const; 
   
   virtual bool selector( Expression* e ); 
   
private:
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );
   static void apply_interactive_( const PStep* self, VMContext* ctx );
   static void apply_rule_( const PStep* self, VMContext* ctx );
   
   Expression* m_expr;

   bool m_bInteractive;
   bool m_bInRule;
};

}

#endif

/* end of stmtautoexpr.h */
