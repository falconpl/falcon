/*
   FALCON - The Falcon Programming Language.
   FILE: exprcompare.cpp

   Expression elements -- Comparisons
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 23:35:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/errors/operanderror.h>

#include <falcon/psteps/exprcompare.h>

namespace Falcon {

// Inline class to simplify
template <class _cpr >
bool generic_simplify( Item& value, Expression* m_first, Expression* m_second )
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( _cpr::pass( d1.asInteger(), d2.asInteger() ) );
         break;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( _cpr::passn( d1.asInteger(), d2.asNumeric() ) );
         break;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( _cpr::passn( d1.asNumeric(), d2.asInteger() ) ) ;
         break;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( _cpr::passn( d1.asNumeric(), d2.asNumeric() ) );
         break;
      default:
         value.setBoolean( _cpr::pass( d1.type(), d2.type() ) );
      }

      return true;
   }

   return false;
}


// Inline class to apply
template <class _cpr >
void generic_apply_( const PStep* ps, VMContext* ctx )
{
   const ExprCompare* self = static_cast<const ExprCompare*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );

   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
   case 0: 
      // check the first operand.
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
      // fallthrough
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->m_second, cf ) )
      {
         return;
      }
      // fallthrough  
   }

   // copy the second
   register Item& op1 = ctx->opcodeParam(1);
   register Item& op2 = ctx->opcodeParam(0);

   switch ( op1.type() << 8 | op2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      op1.setBoolean( _cpr::pass( op1.asInteger(), op2.asInteger() ) );
      ctx->popData();
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      op1.setBoolean( _cpr::pass( op1.asInteger(), op2.asNumeric() ) );
      ctx->popData();
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      op1.setBoolean( _cpr::pass( op1.asNumeric(), op2.asInteger() ) );
      ctx->popData();
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      op1.setBoolean( _cpr::pass( op1.asNumeric(), op2.asNumeric() ) );
      ctx->popData();
      break;

   case FLC_ITEM_USER << 8 | FLC_ITEM_NIL:
   case FLC_ITEM_USER << 8 | FLC_ITEM_BOOL:
   case FLC_ITEM_USER << 8 | FLC_ITEM_INT:
   case FLC_ITEM_USER << 8 | FLC_ITEM_NUM:
   case FLC_ITEM_USER << 8 | FLC_ITEM_METHOD:
   case FLC_ITEM_USER << 8 | FLC_ITEM_FUNC:
   case FLC_ITEM_USER << 8 | FLC_ITEM_USER:
      op1.asClass()->op_compare( ctx, op1.asInst() );
      // refetch, we may have gone deep
      fassert( ctx->topData().isInteger() );
      {
         int64 cmp = ctx->topData().asInteger();
         ctx->popData();
         ctx->topData().setBoolean( _cpr::cmpCheck( cmp ) );
      }
      
      break;

   default:
      op1.setBoolean( _cpr::cmpCheck( op1.compare(op2) ) );
      ctx->popData();
   }
   
   ctx->popCode();
}

template
void generic_apply_<ExprLT::comparer>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprLE::comparer>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprGT::comparer>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprGE::comparer>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprEQ::comparer>( const PStep* ps, VMContext* ctx );

template
void generic_apply_<ExprNE::comparer>( const PStep* ps, VMContext* ctx );

//==========================================================


ExprCompare::ExprCompare( Expression* op1, Expression* op2, Expression::operator_t t, const String& name ):
   BinaryExpression( t, op1, op2 ),
   m_name(name)
{}

ExprCompare::ExprCompare( const ExprCompare& other ):
   BinaryExpression( other ),
   m_name( other.m_name )
{}

ExprCompare::~ExprCompare()
{}



void ExprCompare::describeTo( String& ret ) const
{
   ret = "(" + m_first->describe() + m_name + m_second->describe() + ")";
}


//========================================================
// EXPR LT
//

ExprLT::ExprLT( Expression* op1, Expression* op2 ):
   ExprCompare( op1, op2, t_lt, "<" )
{
   apply = &generic_apply_<ExprLT::comparer>;
}

ExprLT::~ExprLT()
{}

bool ExprLT::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}


//========================================================
// EXPR LE
//
ExprLE::ExprLE( Expression* op1, Expression* op2 ):
   ExprCompare( op1, op2, t_le, "<=" )
{
   apply = &generic_apply_<ExprLE::comparer>;
}


ExprLE::~ExprLE()
{}

bool ExprLE::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

//========================================================
// EXPR GT
//
ExprGT::ExprGT( Expression* op1, Expression* op2 ):
   ExprCompare( op1, op2, t_gt, ">" )
{
   apply = &generic_apply_<ExprGT::comparer>;
}

ExprGT::~ExprGT()
{}

bool ExprGT::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

//========================================================
// EXPR GE
//
ExprGE::ExprGE( Expression* op1, Expression* op2 ):
   ExprCompare( op1, op2, t_ge, ">=" )
{
   apply = &generic_apply_<ExprGE::comparer>;
}

ExprGE::~ExprGE()
{}

bool ExprGE::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

//========================================================
// EXPR EQ
//
ExprEQ::ExprEQ( Expression* op1, Expression* op2 ):
   ExprCompare( op1, op2, t_eq, "==" )
{
   apply = &generic_apply_<ExprEQ::comparer>;
}

ExprEQ::~ExprEQ()
{}

bool ExprEQ::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}


//========================================================
// EXPR NEQ
//
ExprNE::ExprNE( Expression* op1, Expression* op2 ):
   ExprCompare( op1, op2, t_neq, "!=" )
{
   apply = &generic_apply_<ExprNE::comparer>;
}

ExprNE::~ExprNE()
{}


bool ExprNE::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}


}

/* end of exprcompare.cpp */
