/*
   FALCON - The Falcon Programming Language.
   FILE: exprdict.h

   Syntactic tree item definitions -- dictionary of expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 18:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRDICT_H
#define FALCON_EXPRDICT_H

#include <falcon/expression.h>

namespace Falcon
{

/** Expression declaring dictionaries. */
class FALCON_DYN_CLASS ExprDict: public Expression
{
public:
   ExprDict();
   ExprDict( const ExprDict& other );
   virtual ~ExprDict();

   /** Gets the number of sub-expressions in this expression-array.
    \return Count of expressions held in this array.
    */

   size_t arity() const;
   /** Get the nth expression in the array.
    \param n The number of the expression pair that must be accessed.
    \param first The first expression of the pair.
    \param second The second expression of the pair.
    \return true if the expressions can be get, false otherwise.
    */
    bool get( size_t n, Expression* &first, Expression* &second ) const;

   /** Adds another expression to this array.
    \param k The first expression of the new pair.
    \param k The second expression of the new pair.
    \return itself (useful for declarations in sources)
    */
   ExprDict& add( Expression* k, Expression* v );

   virtual void describeTo( String&, int depth=0 ) const;
   virtual void oneLinerTo( String& s ) const;
   
   static void apply_( const PStep*, VMContext* vm );

   inline virtual ExprDict* clone() const { return new ExprDict( *this ); }

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;

private:
   class Private;
   Private* _p;
};

}

#endif

/* end of exprdict.h */
