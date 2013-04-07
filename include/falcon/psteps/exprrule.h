/*
   FALCON - The Falcon Programming Language.
   FILE: exprrule.h

   Syntactic tree item definitions -- Rule expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRRULE_H
#define FALCON_EXPRRULE_H

#include <falcon/statement.h>
#include <falcon/rulesyntree.h>
#include <falcon/expression.h>

namespace Falcon
{

/** Rule statement.

   The rule statement processes one or more sub-trees in a rule context.
*/
class FALCON_DYN_CLASS ExprRule: public Expression
{
public:
   ExprRule( int32 line=0, int32 chr=0 );
   ExprRule( const ExprRule& other );   
   virtual ~ExprRule();
   
   ExprRule& addStatement( TreeStep* stmt );
   ExprRule& addAlternative();

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual ExprRule* clone() const { return new ExprRule(*this); }
   
   virtual bool simplify(Falcon::Item&) const {return false; }
   virtual bool isStandAlone() const {return true; }

   static void apply_( const PStep*, VMContext* ctx );

   SynTree& currentTree();
   const SynTree& currentTree() const;

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool append( TreeStep* element );
   virtual bool remove( int32 pos );

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

   virtual void render( TextWriter* tw, int32 depth ) const;
   
   virtual StmtCut* clone() const { return new StmtCut(*this); }
   
   virtual TreeStep* selector()  const;
   virtual bool selector( TreeStep* expr );
   
private:
   TreeStep* m_expr;
   
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

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual StmtDoubt* clone() const { return new StmtDoubt(*this); }
   
   virtual TreeStep* selector() const;
   virtual bool selector( TreeStep* e );

private:
   TreeStep* m_expr;
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of exprrule.h */
