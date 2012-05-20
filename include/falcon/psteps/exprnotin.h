/* 
 * File:   exprnotin.h
 * Author: francesco
 *
 * Created on 6 maggio 2012, 14.55
 */

#ifndef EXPRNOTIN_H
#define	EXPRNOTIN_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {

/* Notin operator */
class FALCON_DYN_CLASS ExprNotin : public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprNotin, expr_notin )
};
   
}

#endif

/* end of exprnotin.h */

