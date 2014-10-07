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

#undef SRC
#define SRC "engine/psteps/exprcompare.cpp"

#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>
#include <falcon/stderrors.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/psteps/exprcompare.h>

namespace Falcon {

// Inline class to simplify
template <class _cpr >
bool generic_simplify( Item& value, TreeStep* m_first, TreeStep* m_second )
{
   Item d1, d2;
   if(
            m_first->category() == TreeStep::e_cat_expression && static_cast<Expression*>(m_first)->simplify(d1)
         && m_second->category() == TreeStep::e_cat_expression && static_cast<Expression*>(m_second)->simplify(d2) )
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
         break;
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
   
   fassert( self->first() != 0 );
   fassert( self->second() != 0 );

   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
   case 0: 
      // check the first operand.
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->first(), cf ) )
      {
         return;
      }
      /* no break */
   case 1:
      cf.m_seqId = 2;
      if( ctx->stepInYield( self->second(), cf ) )
      {
         return;
      }
      break;
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
      op1.setBoolean( _cpr::passn( op1.asNumeric(), op2.asNumeric() ) );
      ctx->popData();
      break;

   default:
      if( op1.type() >= FLC_ITEM_USER )
      {
         ctx->resetCode( &self->m_stepPostComparer );
         long depth = ctx->codeDepth();

         op1.asClass()->op_compare( ctx, op1.asInst() );

         if( depth != ctx->codeDepth() )
         {
            return;
         }

         // refetch, we may have gone deep
         fassert( ctx->topData().isInteger() );
         int64 cmp = ctx->topData().asInteger();
         ctx->topData().setBoolean( _cpr::cmpCheck( cmp ) );
         // the popCode at the end of the function will take care of stepPostCompare
      }
      else {
         op1.setBoolean( _cpr::cmpCheck( op1.compare(op2) ) );
         ctx->popData();
      }
      break;
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

ExprCompare::ExprCompare( int line, int chr ):
   BinaryExpression( line, chr ),
   m_stepPostComparer(this)
{}

ExprCompare::ExprCompare( Expression* op1, Expression* op2, int line, int chr ):
   BinaryExpression( op1, op2, line, chr ),
   m_stepPostComparer(this)
{}

ExprCompare::ExprCompare( const ExprCompare& other ):
   BinaryExpression( other ),
   m_stepPostComparer(this)
{}

ExprCompare::~ExprCompare()
{}

void ExprCompare::PStepPostCompare::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprCompare::PStepPostCompare* self = static_cast<const ExprCompare::PStepPostCompare*>(ps);
   ctx->topData().setBoolean(self->m_owner->checkCompare( ctx->topData().asInteger() ) );
   ctx->popCode();
}


//========================================================
// EXPR LT
//

ExprLT::ExprLT( int line, int chr ):
   ExprCompare( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_lt )
   apply = &generic_apply_<ExprLT::comparer>;
}

ExprLT::ExprLT( Expression* op1, Expression* op2, int line, int chr ):
   ExprCompare( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_lt )
   apply = &generic_apply_<ExprLT::comparer>;
}

ExprLT::ExprLT( const ExprLT& other ):
   ExprCompare(other)
{
   apply = &generic_apply_<ExprLT::comparer>;
}


ExprLT::~ExprLT()
{}

bool ExprLT::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

const String& ExprLT::exprName() const
{
   static String name("<");
   return name;
}

//========================================================
// EXPR LE
//
ExprLE::ExprLE( int line, int chr ):
   ExprCompare( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_le )
   apply = &generic_apply_<ExprLE::comparer>;
}

ExprLE::ExprLE( Expression* op1, Expression* op2, int line, int chr ):
   ExprCompare( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_le )
   apply = &generic_apply_<ExprLE::comparer>;
}

ExprLE::ExprLE( const ExprLT& other ):
   ExprCompare(other)
{
   apply = &generic_apply_<ExprLE::comparer>;
}

ExprLE::~ExprLE()
{}

bool ExprLE::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

const String& ExprLE::exprName() const
{
   static String name("<=");
   return name;
}

//========================================================
// EXPR GT
//
ExprGT::ExprGT( int line, int chr ):
   ExprCompare( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_gt )
   apply = &generic_apply_<ExprGT::comparer>;
}

ExprGT::ExprGT( Expression* op1, Expression* op2, int line, int chr ):
   ExprCompare( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_gt )
   apply = &generic_apply_<ExprGT::comparer>;
}

ExprGT::ExprGT( const ExprGT& other ):
   ExprCompare(other)
{
   apply = &generic_apply_<ExprGT::comparer>;
}

ExprGT::~ExprGT()
{}

bool ExprGT::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

const String& ExprGT::exprName() const
{
   static String name(">");
   return name;
}

//========================================================
// EXPR GE
//
ExprGE::ExprGE( int line, int chr ):
   ExprCompare( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_ge )
   apply = &generic_apply_<ExprGE::comparer>;
}

ExprGE::ExprGE( Expression* op1, Expression* op2, int line, int chr ):
   ExprCompare( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_ge )
   apply = &generic_apply_<ExprGE::comparer>;
}

ExprGE::ExprGE( const ExprGE& other ):
   ExprCompare(other)
{
   apply = &generic_apply_<ExprGE::comparer>;
}

ExprGE::~ExprGE()
{}

bool ExprGE::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}


const String& ExprGE::exprName() const
{
   static String name(">=");
   return name;
}

//========================================================
// EXPR EQ
//
ExprEQ::ExprEQ( int line, int chr ):
   ExprCompare( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_eq )
   apply = &generic_apply_<ExprEQ::comparer>;
}

ExprEQ::ExprEQ( Expression* op1, Expression* op2, int line, int chr ):
   ExprCompare( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_eq )
   apply = &generic_apply_<ExprEQ::comparer>;
}

ExprEQ::ExprEQ( const ExprEQ& other ):
   ExprCompare(other)
{
   apply = &generic_apply_<ExprEQ::comparer>;
}

ExprEQ::~ExprEQ()
{}

bool ExprEQ::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

const String& ExprEQ::exprName() const
{
   static String name("==");
   return name;
}


//========================================================
// EXPR NEQ
//
ExprNE::ExprNE( int line, int chr ):
   ExprCompare( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_ne )
   apply = &generic_apply_<ExprNE::comparer>;
}

ExprNE::ExprNE( Expression* op1, Expression* op2, int line, int chr ):
   ExprCompare( op1, op2, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_ne )
   apply = &generic_apply_<ExprNE::comparer>;
}

ExprNE::ExprNE( const ExprNE& other ):
   ExprCompare(other)
{
   apply = &generic_apply_<ExprNE::comparer>;
}

ExprNE::~ExprNE()
{}


bool ExprNE::simplify( Item& value ) const
{
   return generic_simplify<comparer>( value, m_first, m_second );
}

const String& ExprNE::exprName() const
{
   static String name("!=");
   return name;
}

}

/* end of exprcompare.cpp */
