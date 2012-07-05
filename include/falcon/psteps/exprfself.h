/*
   FALCON - The Falcon Programming Language.
   FILE: exprfself.h

   Syntactic tree item definitions -- fself -- ref to this function.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 05 Jul 2012 02:03:34 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRFSELF_H_
#define FALCON_EXPRFSELF_H_

#include <falcon/expression.h>

namespace Falcon {

/** Class implementing FSelf ("this function") atom value.
 */
class FALCON_DYN_CLASS ExprFSelf: public Expression
{
public:
   ExprFSelf( int line = 0, int chr = 0 );
   ExprFSelf( const ExprFSelf &other );
   virtual ~ExprFSelf();

   virtual bool isStatic() const;
   virtual ExprFSelf* clone() const;
   virtual bool simplify( Item& result ) const;
   virtual void describeTo( String & str, int depth = 0 ) const;

private:
   static void apply_( const PStep* s1, VMContext* ctx );
};

}

#endif

/* end of exprfself.h */
