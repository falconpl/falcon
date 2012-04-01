/*
   FALCON - The Falcon Programming Language.
   FILE: exprcompose.h

   Syntactic tree item definitions -- Function composition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Apr 2012 15:49:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPR_COMPOSE_H_
#define FALCON_EXPR_COMPOSE_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

/** Function composition.
 
 The expression class has a call operator that propagates its parameters
 to the second operand, and then calls the first with the result of the second.
 
 This operator means that (f ^. g)(x,...) === {x => f(g(x,...))}.
 
 f and g can either be functions or expressions; g must have the same arity of
 the parameters of the composition, while f must be unary.
 */
class FALCON_DYN_CLASS ExprCompose: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprCompose, expr_compose );
};

}

#endif

/* end of exprcompose.h */
