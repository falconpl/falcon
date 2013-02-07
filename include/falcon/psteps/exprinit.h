/*
   FALCON - The Falcon Programming Language.
   FILE: exprinit.h

   Syntactic tree item definitions -- Init values for generators
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 05 Jul 2012 02:03:34 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRINIT_H_
#define FALCON_EXPRINIT_H_

#include <falcon/expression.h>

namespace Falcon {

/** Class implementing init value for generators.
 */
class FALCON_DYN_CLASS ExprInit: public Expression
{
public:
   ExprInit( int line = 0, int chr = 0 );
   ExprInit( const ExprInit &other );
   virtual ~ExprInit();

   virtual bool isStatic() const;
   virtual ExprInit* clone() const { return new ExprInit(*this); }
   virtual bool simplify( Item& result ) const { return false; }
   virtual void describeTo( String & str, int depth = 0 ) const;

private:
   static void apply_( const PStep* s1, VMContext* ctx );
};

}

#endif

/* end of exprinit.h */
