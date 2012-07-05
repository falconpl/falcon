/*
   FALCON - The Falcon Programming Language.
   FILE: exprevalretexec.h

   Syntactic tree item definitions -- Return and eval again from eval
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 05 Jul 2012 15:06:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPREVALRETEXEC_H
#define FALCON_EXPREVALRETEXEC_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** Unary negative. */
class FALCON_DYN_CLASS ExprEvalRetExec: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprEvalRetExec, expr_evalretexec );
};

}

#endif	/* EXPRNEG_H */

/* end of exprevalretexec.h */
