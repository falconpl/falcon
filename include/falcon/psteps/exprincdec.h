/*
   FALCON - The Falcon Programming Language.
   FILE: exprincdec.h

   Increment/decrement expressions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 00:39:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRINCDEC_H_
#define _FALCON_EXPRINCDEC_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon
{

/** Math unary increment prefix. */
class FALCON_DYN_CLASS ExprPreInc: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPreInc, expr_preinc );
   inline virtual bool isStandAlone() const { return true; }
   
private:   
   class ops;
};

/** Math unary increment postfix. */
class FALCON_DYN_CLASS ExprPostInc: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPostInc, expr_postinc );
   inline virtual bool isStandAlone() const { return true; }

private:
   class ops;
};

/** Math unary decrement prefix. */
class FALCON_DYN_CLASS ExprPreDec: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPreDec, expr_predec );
   inline virtual bool isStandAlone() const { return true; }
   
private:
   class ops;
};

/** Math unary decrement postfix. */
class FALCON_DYN_CLASS ExprPostDec: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPostDec, expr_postdec );
   inline virtual bool isStandAlone() const { return true; }

private:
   class ops;
};

}

#endif

/* end of exprincdec.h */
