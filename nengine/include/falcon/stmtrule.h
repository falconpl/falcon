/*
   FALCON - The Falcon Programming Language.
   FILE: stmtrule.h

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTRULE_H
#define FALCON_STMTRULE_H

#include <falcon/statement.h>
#include <falcon/rulesyntree.h>
#include <falcon/expression.h>
#include <falcon/pcode.h>

namespace Falcon
{

/** Rule statement.

   The rule statement processes one or more sub-trees in a rule context.
*/
class FALCON_DYN_CLASS StmtRule: public Statement
{
public:
   StmtRule( int32 line=0, int32 chr=0 );
   virtual ~StmtRule();
   
   StmtRule& addStatement( Statement* stmt );
   StmtRule& addAlternative();

   void describeTo( String& tgt ) const;

   static void apply_( const PStep*, VMContext* ctx );

   SynTree& currentTree();
   const SynTree& currentTree() const;
   
protected:
   class Private;
   Private* _p;
};

/** Cut statement.
   Kills the current rule context in a rule, or makes an expression
  "deterministic".
*/
class FALCON_DYN_CLASS StmtCut: public Statement
{
public:
   StmtCut( Expression* expr = 0, int32 line=0, int32 chr=0 );
   virtual ~StmtCut();

   void describeTo( String& tgt ) const;

private:
   Expression* m_expr;
   PCode m_pc;
   
   static void apply_( const PStep*, VMContext* ctx );
   static void apply_cut_expr_( const PStep*, VMContext* ctx );
};


/** Doubt statement.
 Forces an expression to be non-deterministic.
*/
class FALCON_DYN_CLASS StmtDoubt: public Statement
{
public:
   StmtDoubt( Expression* expr, int32 line=0, int32 chr=0 );
   virtual ~StmtDoubt();

   void describeTo( String& tgt ) const;

private:
   Expression* m_expr;
   PCode m_pc;
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtrule.h */
