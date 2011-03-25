/*
   FALCON - The Falcon Programming Language.
   FILE: exprfactory.cpp

   Syntactic tree item definitions -- expression factory.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/closedsymbol.h>
#include <falcon/globalsymbol.h>
#include <falcon/localsymbol.h>
#include <falcon/dynsymbol.h>

#include <falcon/exprfactory.h>
#include <falcon/datareader.h>

namespace Falcon {

Expression* ExprFactory::make( Expression::operator_t type )
{
   switch( type )
   {

   case Expression::t_value: return new ExprValue;
   case Expression::t_neg: return new ExprNeg;
   case Expression::t_not: return new ExprNot;

   case Expression::t_and: return new ExprAnd;
   case Expression::t_or: return new ExprOr;

   case Expression::t_plus: return new ExprPlus;

   /*
       * TODO
   case Expression::t_minus: return new ExprMinus;
   case Expression::t_times: return new ExprTimes;
   case Expression::t_divide: return new ExprDiv;
   case Expression::t_modulo: return new ExprMod;
   case Expression::t_power: return new ExprPow;

   case Expression::t_pre_inc: return new ExprPreInc;
   case Expression::t_post_inc: return new ExprPostInc;
   case Expression::t_pre_dec: return new ExprPreDec;
   case Expression::t_post_dec: return new ExprPostDec;

   case Expression::t_gt: return new ExprGT;
   case Expression::t_ge: return new ExprGE;
   case Expression::t_lt: return new ExprLT;
   case Expression::t_le: return new ExprLE;
   case Expression::t_eq: return new ExprEQ;
   case Expression::t_exeq: return new ExprEEQ;
   case Expression::t_neq: return new ExprNE;

   case Expression::t_in: return new ExprIn;
   case Expression::t_notin: return new ExprNotIn;
   case Expression::t_provides: return new ExprProvides;

   case Expression::t_iif: return new ExprIIF;

   case Expression::t_obj_access: return new ExprDot;
   case Expression::t_funcall: return new ExprCall;
   case Expression::t_array_access: return new ExprIndex;
   case Expression::t_array_byte_access: return new ExprStarIndex;
   case Expression::t_strexpand: return new ExprStrExpand;
   case Expression::t_indirect: return new ExprIndirect;

   case Expression::t_assign: return new ExprAssign;
   case Expression::t_fbind: return new ExprFbind;

   case Expression::t_aadd: return new ExprAutoAdd;
   case Expression::t_asub: return new ExprAutoSub;
   case Expression::t_amul: return new ExprAutoMul;
   case Expression::t_adiv: return new ExprAutoDiv;
   case Expression::t_amod: return new ExprAutoMod;
   case Expression::t_apow: return new ExprAutoPow;

   case Expression::t_eval: return new ExprEval;
   case Expression::t_oob: return new ExprOob;
   case Expression::t_deoob: return new ExprDeoob;
   case Expression::t_xoroob: return new ExprXorOob;
   case Expression::t_isoob: return new ExprIsOob;
   */
   }
   return 0;
}


Expression* ExprFactory::deserialize( DataReader* s )
{
   byte b;
   s->read( b );
   Expression::operator_t type = (Expression::operator_t)( b );

   Expression* expr = make( type );
   if ( expr == 0 )
   {
      //throw new IoError(ErrorParam( e_deser, __LINE__ ).extra( "Expression.deserialize"));
   }

   try {
      expr->deserialize( s );
      return expr;
   }
   catch( ... )
   {
      delete expr;
      throw;
   }
}

}

/* end of exprfactory.cpp */
