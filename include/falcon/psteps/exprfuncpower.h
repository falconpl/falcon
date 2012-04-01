/*
   FALCON - The Falcon Programming Language.
   FILE: exprfuncpower.h

   Syntactic tree item definitions -- Function power
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Apr 2012 15:49:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPR_FUNCPOWER_H_
#define FALCON_EXPR_FUNCPOWER_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

/** Function composition.
 
 The expression calls the function or evaluates the expression a number of times
 equal to the second operand (which is evaluated first).
 
 This operator means that (f ^.. n)(x) === {x => f(f(f...(x) ...))}.
 
 f must be unary.
 */
class FALCON_DYN_CLASS ExprFuncPower: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprFuncPower, expr_funcpower );
};

}

#endif

/* end of exprfuncpower.h */
