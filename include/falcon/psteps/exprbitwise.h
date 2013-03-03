/*
   FALCON - The Falcon Programming Language.
   FILE: exprbitwise.h

   Syntactic tree item definitions -- Bitwise expressions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 13:22:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRBITWISE_H
#define FALCON_EXPRBITWISE_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

class FALCON_DYN_CLASS ExprBNOT: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprBNOT, expr_bnot );
};

}

#endif	/* EXPRBITWISE_H */

/* end of exprbitwise.h */
