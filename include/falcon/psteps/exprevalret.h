/*
   FALCON - The Falcon Programming Language.
   FILE: exprevalret.h

   Syntactic tree item definitions -- Return from evaluation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 05 Jul 2012 15:06:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPREVALRET_H
#define FALCON_EXPREVALRET_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** Unary negative. */
class FALCON_DYN_CLASS ExprEvalRet: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprEvalRet, expr_evalret );
};

}

#endif	/* EXPRNEG_H */

/* end of exprevalret.h */
