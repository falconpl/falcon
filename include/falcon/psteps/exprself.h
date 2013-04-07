/*
   FALCON - The Falcon Programming Language.
   FILE: exprself.h

   Syntactic tree item definitions -- Self accessor expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 13:46:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRSELF_H_
#define FALCON_EXPRSELF_H_

#include <falcon/expression.h>

namespace Falcon {

/** Class implementing Self atom value.
 */
class FALCON_DYN_CLASS ExprSelf: public Expression
{
public:
   ExprSelf( int line = 0, int chr = 0 );
   ExprSelf( const ExprSelf &other );
   virtual ~ExprSelf();

   virtual ExprSelf* clone() const;
   virtual void render( TextWriter* tw, int32 depth ) const;

private:
   static void apply_( const PStep* s1, VMContext* ctx );

};

}

#endif

/* end of exprself.h */
