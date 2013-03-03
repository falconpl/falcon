/*
   FALCON - The Falcon Programming Language.
   FILE: expreeq.h

   Syntactic tree item definitions -- Exactly equal (===) expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPR_EEQ_H_
#define FALCON_EXPR_EEQ_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

/** Exactly equal to operator. */
class FALCON_DYN_CLASS ExprEEQ: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprEEQ, expr_eeq );
};

}

#endif

/* end of expreeq.h */
