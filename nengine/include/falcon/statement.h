/*
   FALCON - The Falcon Programming Language.
   FILE: statement.h

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STATEMENT_H
#define FALCON_STATEMENT_H

#include <falcon/pstep.h>
#include <falcon/pcode.h>
#include <falcon/vm.h>

namespace Falcon
{

class Expression;
class SynTree;

/** Statement.
 * Statements are PStep that may require other sub-sequences to be evaluated.
 * In other words, they are
 */
class Statement: public PStep
{

public:
   typedef enum {
      autoexpr_t,
      if_t,
      while_t
   } statement_t ;

   Statement( statement_t type ):
      m_type(type)
   {}

   inline virtual ~Statement() {}

protected:
   /** Steps being prepared by the statement */
   std::vector<PStep*> m_steps;

   inline void prepare( VMachine* vm ) const
   {
      std::vector<PStep*>::const_iterator b = m_steps.begin();
      while( b != m_steps.end() ){
         vm->pushCode(*b);
         ++b;
      }
   }

   friend class SynTree;
private:
   statement_t m_type;
};

/** Autoexpression.
 *
 * This statement is needed to wrap an expression so that its
 * result is removed from the stack.
 *
 * The obvious un-necessity of this class advises for the usage of
 * registers where expressions place their results; but reasons
 * against are equally valid.
 */
class StmtAutoexpr: public Statement
{
public:
   StmtAutoexpr( Expression* expr );
   virtual ~StmtAutoexpr();

   void toString( String& tgt ) const;
   virtual void apply( VMachine* vm ) const;

private:
   Expression* m_expr;
   PCode m_pcExpr;
};


class StmtWhile: public Statement
{
public:
   StmtWhile( Expression* check, SynTree* stmts );
   virtual ~StmtWhile();

   void toString( String& tgt ) const;
   virtual void apply( VMachine* vm ) const;

private:
   Expression* m_check;
   PCode m_pcCheck;
   SynTree* m_stmts;
};


class StmtIf: public Statement
{
public:
   StmtIf( Expression* check, SynTree* ifTrue, SynTree* ifFalse = 0 );
   virtual ~StmtIf();

   void toString( String& tgt ) const;
   virtual void apply( VMachine* vm ) const;

   /** Adds an else-if branch to the if statement */
   StmtIf& addElif( Expression *check, SynTree* ifTrue );

   /** Sets the else branch for this if statement. */
   StmtIf& setElse( SynTree* ifFalse );

private:
   SynTree* m_ifFalse;

   class ElifBranch
   {
   public:
      Expression* m_check;
      PCode m_pcCheck;
      SynTree* m_ifTrue;

      ElifBranch( Expression *check, SynTree* ifTrue ):
         m_check( check ),
         m_ifTrue( ifTrue )
      {}

      ~ElifBranch();

      void compile();
   };

   std::vector<ElifBranch> m_elifs;
};

}

#endif

/* end of statement.h */

