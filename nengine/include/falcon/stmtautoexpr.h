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
   StmtAutoexpr( Expression* expr, int32 line=0, int32 chr = 0 );
   virtual ~StmtAutoexpr();

   void describe( String& tgt ) const;
   inline String describe() const { return PStep::describe(); }

   void oneLiner( String& tgt ) const;
   inline String oneLiner() const { return PStep::oneLiner(); }

   /** Check explicit non-determinism set. 
    If true, then an explicit "?" is specified for this statement.
   */
   bool nd() const { return m_nd; }
   
   /** Sets explicit non-determinism status.
    \parm mode If this statement must be forced to be non-deterministic.
    \throw CodeError if determ() is already set.
    In rules, the "?" prefix indicates a rule statement that may fail.
   */
   void nd( bool mode );

   /** Check if explicit determinism is set.
    If true, then an explicit "*" is specified for this statement.
   */
   bool determ() const { return m_determ; }

   /** Sets explicit determinism status.
    \parm mode If this statement must be forced to be deterministic.
    \throw CodeError if nd() is already set.
    In rules, the "?" prefix indicates a rule statement that may fail.
   */
   void determ( bool mode );

   /** Returns the expression held by this expression-statement.
    \return The held expression, or 0 if it was not set.
    */
   Expression* expr() const { return m_expr; }

   /** Removes the expression stored in this AutoExpression.
    \return The held expression, or 0 if it was not set.

    This method can be used when the parser generated an autoexpression
    that is actually used elsewhere.
    */
   Expression* detachExpr() {
      Expression* expr = m_expr;
      m_expr = 0;
      return expr;
   }
   
private:
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );
   
   Expression* m_expr;
   PCode m_pcExpr;

   bool m_nd;
   bool m_determ;
};

}

#endif

/* end of stmtautoexpr.h */
