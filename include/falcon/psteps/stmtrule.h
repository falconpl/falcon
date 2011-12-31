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

namespace Falcon
{

/** Rule statement.

   The rule statement processes one or more sub-trees in a rule context.
*/
class FALCON_DYN_CLASS StmtRule: public Statement
{
public:
   StmtRule( int32 line=0, int32 chr=0 );
   StmtRule( const StmtRule& other );   
   virtual ~StmtRule();
   
   StmtRule& addStatement( Statement* stmt );
   StmtRule& addAlternative();

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtRule* clone() const { return new StmtRule(*this); }
   
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
   StmtCut( int32 line=0, int32 chr=0 );
   StmtCut( Expression* expr, int32 line=0, int32 chr=0 );
   StmtCut( const StmtCut& expr );   
   virtual ~StmtCut();

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   
   virtual StmtCut* clone() const { return new StmtCut(*this); }
   
   virtual Expression* selector()  const;
   virtual bool selector( Expression* expr );
   
private:
   Expression* m_expr;
   
   static void apply_( const PStep*, VMContext* ctx );
   static void apply_cut_expr_( const PStep*, VMContext* ctx );
};


/** Doubt statement.
 Forces an expression to be non-deterministic.
*/
class FALCON_DYN_CLASS StmtDoubt: public Statement
{
public:
   StmtDoubt( int32 line=0, int32 chr=0 );
   StmtDoubt( Expression* expr, int32 line=0, int32 chr=0 );
   StmtDoubt( const StmtDoubt& other );
   virtual ~StmtDoubt();

   virtual void describeTo( String& tgt, int depth=0) const;
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtDoubt* clone() const { return new StmtDoubt(*this); }
   
private:
   Expression* m_expr;
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtrule.h */
