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

#include <falcon/psteps/exprvector.h>

namespace Falcon
{

/** Expression declaring dictionaries. 
 \TODO: plublish a method to insert a parir.
 \note The remove method removes a pair of elements.
 */
class FALCON_DYN_CLASS ExprDict: public ExprVector
{
public:
   ExprDict(int line=0, int chr=0);
   ExprDict( const ExprDict& other );
   virtual ~ExprDict();
   
   /** Overridden to forbid acceptance of single elements. */
   virtual bool insert( int32 pos, TreeStep* element );
   
   /** Will remove a pair of element. Pos is the nth pair. */
   virtual bool remove( int32 pos );
   
   /** Return the number of pairs in the dictionary.
    
    Equals to arity / 2
    */
   int pairs() const;
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

};

}

#endif

/* end of exprdict.h */
