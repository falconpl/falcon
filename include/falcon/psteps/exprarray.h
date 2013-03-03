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

#include <falcon/psteps/exprvector.h>

namespace Falcon
{

/** Expression declaring arrays or list of expressions.

 This class is used both to store array declarations and list of expressions,
 for example when generating parametric calls or assignment lists.

 */
class FALCON_DYN_CLASS ExprArray: public ExprVector
{
public:
   ExprArray( int line = 0, int chr = 0);
   ExprArray( const ExprArray& other );  
   
   /** Gets the number of sub-expressions in this expression-array.
    \return Count of expressions held in this array.
    */

   virtual ~ExprArray() {}

   virtual void describeTo( String& s, int depth=0 ) const;
   virtual void oneLinerTo( String& s ) const;

   static void apply_( const PStep*, VMContext* ctx );

   inline virtual ExprArray* clone() const { return new ExprArray( *this ); }

   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;

};

}

#endif

/* end of exprarray.h */
