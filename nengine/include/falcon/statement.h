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
#include <falcon/syntree.h>

namespace Falcon
{

class VMachine;
class Expression;

/** Statement.
 * Statements are PStep that may require other sub-sequences to be evaluated.
 * In other words, they are
 */
class Statement: public PStep
{

public:
   typedef enum {
      if_t,
      while_t
   } statement_t ;

   Statement( statement_t type ):
      m_type(t)
   {}

   inline virtual ~Statement() {}

private:
   statement_t m_type;
};


class StmtWhile: public Statement
{
public:
   StmtWhile( Expression* check, SynTree* stmts );
   virtual ~StmtWhile();

   void toString( String& tgt ) const;
   virtual void perform( VMachine* vm ) const;
   virtual void apply( VMachine* vm ) const;

private:
   Expression* m_check;
   SynTree* m_stmts;
};


class StmtIf: public Statement
{
public:
   StmtIf( Expression* check, SynTree* ifTrue, SynTree* ifFalse );
   vitual ~StmtIf();

   void toString( String& tgt ) const;
   virtual void perform( VMachine* vm ) const;
   virtual void apply( VMachine* vm ) const;

private:
   Expression* m_check;
   SynTree* m_ifTrue;
   SynTree* m_ifFalse;
};

}

#endif

/* end of statement.h */

