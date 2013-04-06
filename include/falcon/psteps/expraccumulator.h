/*
   FALCON - The Falcon Programming Language.
   FILE: expraccumulator.h

   Syntactic tree item definitions -- Accumulator ^[...] f => tg
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Mar 2013 14:00:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRACCUMULATOR_H
#define FALCON_EXPRACCUMULATOR_H

#include <falcon/expression.h>
#include <falcon/requirement.h>

namespace re2 {
   class RE2;
}

namespace Falcon {
class Symbol;
class DataReader;
class DataWriter;
class CaseRequirement;
class Engine;

/** Accumulator expression.
   The accumulator expression is:

   ^[ a,b,c ] f => tg

 */
class FALCON_DYN_CLASS ExprAccumulator: public Expression
{
public:
   ExprAccumulator( int line =0, int chr = 0 );
   ExprAccumulator( const ExprAccumulator& other );
   virtual ~ExprAccumulator();

   inline virtual ExprAccumulator* clone() const { return new ExprAccumulator( *this ); }
   virtual bool simplify( Item& value ) const;
   bool isStatic() const { return false; }
   inline virtual bool isStandAlone() const { return true; }

   virtual void render( TextWriter* tw, int depth ) const;
   static void apply_( const PStep*, VMContext* ctx );

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );

   virtual Expression* selector() const;
   virtual bool selector( Expression* e );

   bool filter( TreeStep* ts );
   TreeStep* filter() const { return m_filter; }

   bool target( TreeStep* ts );
   TreeStep* target() const { return m_target; }
private:
   TreeStep* m_target;
   TreeStep* m_filter;

   Expression* m_vector;

   FALCON_DECLARE_INTERNAL_PSTEP_OWNED( GenIter, ExprAccumulator );
   FALCON_DECLARE_INTERNAL_PSTEP_OWNED( GenNext, ExprAccumulator );
   FALCON_DECLARE_INTERNAL_PSTEP_OWNED( TakeNext, ExprAccumulator );
   FALCON_DECLARE_INTERNAL_PSTEP_OWNED( AfterFilter, ExprAccumulator );
   FALCON_DECLARE_INTERNAL_PSTEP_OWNED( AfterAddTarget, ExprAccumulator );

   void addToTarget( VMContext* ctx, Item* base, int32 arity ) const;
   void regress( VMContext* ctx ) const;
};

}

#endif

/* end of expraccumulator.h */
