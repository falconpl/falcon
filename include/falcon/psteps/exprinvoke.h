/*
   FALCON - The Falcon Programming Language.
   FILE: exprinvoke.h

   Syntactic tree item definitions -- Invoke expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Jan 2013 16:23:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRINVOKE_H
#define FALCON_EXPRINVOKE_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/** Invoke expression
 * This expression causes the invocation of the first operand by
 * the means of the parametric evaluation through the second operand.
 * \code
 *    func # param  // ==> func(param)
 * \endcode
 *
 * The expression is particolarly useful if the second operand is a
 * evaluation parameter expression (epex), as the parameters are passed
 * directly into the invocation:
 *
 * \code
 *   func # ^(a,b)   // ==> func(a,b)
 * \endcode
 */
class FALCON_DYN_CLASS ExprInvoke: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprInvoke, expr_invoke );
   inline virtual bool isStandAlone() const { return true; }
};

}

#endif

/* end of exprinvoke.h */
