/*
   FALCON - The Falcon Programming Language.
   FILE: exprassign.h

   Assign expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 11:44:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRASSIGN_H
#define FALCON_EXPRASSIGN_H

#include <falcon/expression.h>

namespace Falcon {
/** Assignment operation. */
class FALCON_DYN_CLASS ExprAssign: public BinaryExpression
{
public:
   ExprAssign( int line =0, int chr = 0 );
   ExprAssign( Expression* op1, Expression* op2, int line =0, int chr = 0 );
   ExprAssign( const ExprAssign& other );
   
   inline virtual ExprAssign* clone() const { return new ExprAssign( *this ); }

   virtual bool simplify( Item& value ) const;
   virtual void render( TextWriter* tw, int32 depth ) const;

   inline virtual bool isStandAlone() const { return true; }
   virtual const String& exprName() const;

   /** Reimplemented to prevent setting of non-assignable expressions.
    *
    */
   virtual bool setNth( int32 n, TreeStep* ts );
protected:
   static void apply_( const PStep* ps, VMContext* ctx );
};

}

#endif	/* EXPRASSIGN_H */

/* end of exprassign.h */
