/* 
 * File:   exprin.h
 * Author: francesco
 *
 * Created on 6 maggio 2012, 14.54
 */

#ifndef EXPRIN_H
#define	EXPRIN_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon {
   
/* In operator */
class FALCON_DYN_CLASS ExprIn : public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprIn, expr_in );
};
   
}


#endif

/* end of exprin.h */

