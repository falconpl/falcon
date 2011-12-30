/*
   FALCON - The Falcon Programming Language.
   FILE: exprneg.cpp

   Syntactic tree item definitions -- Numeric unary negator
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 13:22:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRNEG_H
#define FALCON_EXPRNEG_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** Unary negative. */
class FALCON_DYN_CLASS ExprNeg: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprNeg, expr_neg );
};

}

#endif	/* EXPRNEG_H */

/* end of exprneg.h */
