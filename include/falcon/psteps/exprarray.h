/*
   FALCON - The Falcon Programming Language.
   FILE: exprarray.h

   Syntactic tree item definitions -- array (of) expression(s).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 18:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRARRAY_H
#define FALCON_EXPRARRAY_H

#include <falcon/expression.h>

namespace Falcon
{

/** Expression declaring arrays or list of expressions.

 This class is used both to store array declarations and list of expressions,
 for example when generating parametric calls or assignment lists.

 */
class FALCON_DYN_CLASS ExprArray: public Expression
{
public:
   ExprArray();
   ExprArray( const ExprArray& other );

   /** Gets the number of sub-expressions in this expression-array.
    \return Count of expressions held in this array.
    */

   size_t arity() const;
   /** Get the nth expression in the array.
    \param n The number of the expression that must be accessed.
    \return the nth expression or 0 if n is out of range.
    */
   Expression* get( size_t n ) const;

   /** Adds another expression to this array.
    \return itself (useful for declarations in sources)
    */
   ExprArray& add( Expression* e );

   virtual ~ExprArray();

   virtual void serialize( DataWriter* s ) const;
   virtual void precompile( PCode* pcd ) const;

   virtual void describeTo( String& ) const;
   virtual void oneLinerTo( String& s ) const;

   static void apply_( const PStep*, VMContext* ctx );

   inline virtual ExprArray* clone() const { return new ExprArray( *this ); }

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;

protected:
   virtual void deserialize( DataReader* s );

private:
   class Private;
   Private* _p;
};

}

#endif

/* end of exprarray.h */
