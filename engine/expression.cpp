/*
   FALCON - The Falcon Programming Language.
   FILE: expression.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcode.h>
#include <falcon/exprfactory.h>
#include <falcon/operanderror.h>
#include <falcon/codeerror.h>
#include <falcon/trace.h>
#include <falcon/pseudofunc.h>

namespace Falcon {

Expression::Expression( const Expression &other ):
   PStep(other),
   m_pstep_lvalue(0),
   m_operator( other.m_operator ),
   m_sourceRef( other.m_sourceRef )
{}

Expression::~Expression()
{}

void Expression::serialize( DataWriter* s ) const
{
   byte type = (byte) m_operator;
   s->write( type );
   m_sourceRef.serialize( s );
}

void Expression::deserialize( DataReader* s )
{
   m_sourceRef.deserialize( s );
}

void Expression::precompile( PCode* pcode ) const
{
   pcode->pushStep( this );
}

void Expression::precompileLvalue( PCode* pcode ) const
{
   // We do this, but the parser should have blocked us...
   pcode->pushStep( this );
}

void Expression::precompileAutoLvalue( PCode* pcode, const PStep* activity, bool, bool ) const
{
   // We do this, but the parser should have blocked us...
   precompile( pcode );             // access -- prepare params
   // no save
   pcode->pushStep( activity );     // action
   // no restore
   precompileLvalue( pcode );       // storage -- if applicable.
}

//=============================================================

UnaryExpression::UnaryExpression( const UnaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() )
{
}

UnaryExpression::~UnaryExpression()
{
   delete m_first;
}


void UnaryExpression::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling un-exp: %p (%s)", pcode, describe().c_ize() );
   m_first->precompile( pcode );
   pcode->pushStep( this );
}

void UnaryExpression::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
}

void UnaryExpression::deserialize( DataReader* s )
{
   Expression::deserialize(s);
   m_first = ExprFactory::deserialize( s );
}

bool UnaryExpression::isStatic() const
{
   return m_first->isStatic();
}

//=============================================================

BinaryExpression::BinaryExpression( const BinaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() )
{
}

BinaryExpression::~BinaryExpression()
{
   delete m_first;
   delete m_second;
}


void BinaryExpression::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling bin-exp: %p (%s)", pcode, describe().c_ize() );
   m_first->precompile( pcode );
   m_second->precompile( pcode );
   pcode->pushStep( this );
}

void BinaryExpression::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
   m_second->serialize( s );
}

void BinaryExpression::deserialize( DataReader* s )
{
   Expression::deserialize(s);
   m_first = ExprFactory::deserialize( s );
   m_second = ExprFactory::deserialize( s );
}

bool BinaryExpression::isStatic() const
{
   return m_first->isStatic() && m_second->isStatic();
}

//=============================================================
TernaryExpression::TernaryExpression( const TernaryExpression &other ):
   Expression( other ),
   m_first( other.m_first->clone() ),
   m_second( other.m_second->clone() ),
   m_third( other.m_third->clone() )
{
}

TernaryExpression::~TernaryExpression()
{
   delete m_first;
   delete m_second;
   delete m_third;
}

void TernaryExpression::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling tri-exp: %p (%s)", pcode, describe().c_ize() );
   
   m_first->precompile( pcode );
   m_second->precompile( pcode );
   m_third->precompile( pcode );
   pcode->pushStep( this );
}


void TernaryExpression::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   m_third->serialize( s );
   m_second->serialize( s );
   m_first->serialize( s );
}

void TernaryExpression::deserialize( DataReader* s )
{
   Expression::deserialize(s);
   m_first = ExprFactory::deserialize( s );
   m_second = ExprFactory::deserialize( s );
   m_third = ExprFactory::deserialize( s );
}

bool TernaryExpression::isStatic() const
{
   return m_first->isStatic() && m_second->isStatic() && m_third->isStatic();
}


//==================================================================
// Expressions
//

// ===================== logic not.

bool ExprNot::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setBoolean( ! value.isTrue() );
      return true;
   }
   return false;
}

void ExprNot::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprNot*)self)->describe().c_ize() );

   Item& operand = ctx->topData();

   //TODO: overload not
   operand.setBoolean( ! operand.isTrue() );
}

void ExprNot::describeTo( String& str ) const
{
   str = "not ";
   str += m_first->describe();
}

//=========================================================
//Logic And


bool ExprAnd::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      value.setBoolean( fi.isTrue() && si.isTrue() );
      return true;
   }
   return false;

}

void ExprAnd::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );

   // check the first expression for first...
   m_first->precompile( pcode );
    // add a gate to jump checks on short circuits
   pcode->pushStep( &m_gate );
   // and then the second expr last
   m_second->precompile( pcode );
   pcode->pushStep( this );
   //set shortcircuit jump location to end of expr.
   m_gate.m_shortCircuitSeqId = pcode->size();
  
   
}

void ExprAnd::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprAnd*)self)->describe().c_ize() );

   // use the space left from us by the previous expression
   Item& operand = ctx->topData();

   // Booleanize it
   operand.setBoolean( operand.isTrue() );
}

void ExprAnd::describeTo( String& str ) const
{
   str = "(" + m_first->describe() + " and " + m_second->describe() + ")";
}


ExprAnd::Gate::Gate() {
   apply = apply_;
}


void ExprAnd::Gate::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprAnd::Gate*)ps)->describe().c_ize() );

   // read and recycle the topmost data.
   Item& operand = ctx->topData();

   if( ! operand.isTrue() )
   {
      operand.setBoolean( false );
      // pop ourselves and the calling code
      ctx->currentCode().m_seqId = static_cast<const Gate*>(ps)->m_shortCircuitSeqId;
   }
   else {
      // the other expression is bound to push data, remove ours
      ctx->popData();
   }   
}

//=========================================================
//Logic Or

bool ExprOr::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      value.setBoolean( fi.isTrue() || si.isTrue() );
      return true;
   }
   return false;
}


void ExprOr::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );

   // check the first expression for first...
   m_first->precompile( pcode );
   // add a gate to jump checks on short circuits
   pcode->pushStep( &m_gate );
   // and then the second expr last
   m_second->precompile( pcode );
   pcode->pushStep( this );
   //set shortcircuit jump location to end of expr.
   m_gate.m_shortCircuitSeqId = pcode->size();
}


void ExprOr::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprOr*)self)->describe().c_ize() );
   
   // reuse the operand left by the other expression
   Item& operand = ctx->topData();
   operand.setBoolean( operand.isTrue() );
}


void ExprOr::describeTo( String& str ) const
{
   str = "(" + m_first->describe() + " or " + m_second->describe() + ")";
}


ExprOr::Gate::Gate() {
   apply = apply_;
}


void ExprOr::Gate::apply_( const PStep* ps,  VMContext* ctx )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprOr::Gate*)ps)->describe().c_ize() );

   // read and recycle the topmost data.
   Item& operand = ctx->topData();
   if( operand.isTrue() )
   {
      operand.setBoolean( true );
      // pop ourselves and the calling code
      ctx->currentCode().m_seqId = static_cast<const Gate*>(ps)->m_shortCircuitSeqId;
   }
   else {
      // the other expression is bound to push data, remove ours
      ctx->popData();
   }
}

//=========================================================
//Assignment

void ExprAssign::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling Assign: %p (%s)", pcode, describe().c_ize() );

   // just, evaluate the second, then evaluate the first,
   // but the first knows it's a lvalue.
   m_second->precompile(pcode);
   m_first->precompileLvalue(pcode);
}


bool ExprAssign::simplify( Item& ) const
{
   // TODO Simplify for closed symbols
   return false;
}

void ExprAssign::describeTo( String& str ) const
{
   str = "(" + m_first->describe() + " = " + m_second->describe() + ")";
}

//=========================================================
//Math

//=========================================================
//Unary

bool ExprNeg::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( -value.asInteger() ); return true;
      case FLC_ITEM_NUM: value.setNumeric( -value.asNumeric() ); return true;
      }
   }

   return false;
}

void ExprNeg::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprNeg*)self)->describe().c_ize() );
   
   Item& item = ctx->topData();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( -item.asInteger() ); break;
      case FLC_ITEM_NUM: item.setNumeric( -item.asNumeric() ); break;
      case FLC_ITEM_USER:
         item.asClass()->op_neg( ctx, item.asInst() );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("neg") );
   }
}

void ExprNeg::describeTo( String& str ) const
{
   str = "-";
   str += m_first->describe();
}




//=========================================================
//Comparisons

void ExprEEQ::apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprEEQ*)ps)->describe().c_ize() );

   Item *op1, *op2;
   ctx->operands( op1, op2 );

   switch ( op1->type() << 8 | op2->type() )
   {
   case FLC_ITEM_NIL << 8 | FLC_ITEM_NIL:
      op1->setBoolean( true );
      break;

   case FLC_ITEM_BOOL << 8 | FLC_ITEM_BOOL:
      op1->setBoolean( op1->asBoolean() == op2->asBoolean() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      op1->setBoolean( op1->asInteger() == op2->asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      op1->setBoolean( op1->asNumeric() == op2->asNumeric() );
      break;

   case FLC_ITEM_USER << 8 | FLC_ITEM_USER:
      op1->setBoolean( op1->asInst() == op2->asInst() );
      break;

   default:
      op1->setBoolean(false);
   }
   
   ctx->popData();
}


void ExprEEQ::describeTo( String& ret ) const
{
   ret = "(" + m_first->describe() + " === " + m_second->describe() + ")";
}

bool ExprEEQ::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      value.setBoolean( d1.compare(d2) == 0 );
      return true;
   }
   
   return false;
}


//=========================================================
//Fast if (ternary conditional operator)

bool ExprIIF::simplify( Item& value ) const
{
   Item temp;
   if( m_first->simplify( temp ) )
   {
      if (temp.isTrue())
      {
         return m_second->simplify( value );
      }
      else
      {
         return m_third->simplify( value );
      }
   }

   return false;
}


void ExprIIF::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );
   
   //precompile condition executed first.
   m_first->precompile( pcode );
   //push ourselves to determine where to branch to.
   pcode->pushStep( this );
   //precompile true expr.
   m_second->precompile( pcode );
   //push gate to allow exit after true expr.
   pcode->pushStep( &m_gate );
   //acuire the position of the start of the false expr to jump to.
   m_falseSeqId = pcode->size();
   //precompile false expr.
   m_third->precompile( pcode );
   //acquire the position at the end of the expr to jump over the false expr.
   m_gate.m_endSeqId = pcode->size();
   
}

void ExprIIF::apply_( const PStep* self, VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprIIF*)self)->describe().c_ize() );
   
   //get the value of the condition and pop it.
   Item cond = ctx->topData();
   ctx->popData();

   if ( !cond.isTrue() )
   {
      ctx->currentCode().m_seqId = ((ExprIIF*)self)->m_falseSeqId;
   }

   
}

void ExprIIF::describeTo( String& str ) const
{
   str = "( " + m_first->describe() + " ? " + m_second->describe() + " : " + m_third->describe() + " )";
}

ExprIIF::Gate::Gate() {
   apply = apply_;
}

void ExprIIF::Gate::apply_( const PStep* ps,  VMContext* ctx )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprIIF::Gate*)ps)->describe().c_ize() );

   ctx->currentCode().m_seqId = ((ExprIIF::Gate*)ps)->m_endSeqId;
}


//==================================================================
//
//

void ExprStarIndex::apply_( const PStep* DEBUG_ONLY(ps), VMContext* ctx )
{
   TRACE2( "Apply \"%s\"", ((ExprStarIndex*)ps)->describe().c_ize() );

   Item str = ctx->topData();
   ctx->popData();
   Item& index = ctx->topData();

   if ( str.isString() && index.isOrdinal() )
   {
      index.setInteger( str.asString()->getCharAt((length_t)index.forceInteger()) );
   }
   else
   {
      // no need to throw, we're going to get back in the VM.
      throw
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("[*]") );
   }
  
}


void ExprStarIndex::describeTo( String& ret ) const
{
   ret = "(" + m_first->describe() + "[*" + m_second->describe() + "])";
}

//=========================================================
//Oob Manipulators

bool ExprOob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setOob();
      return true;
   }

   return false;
}

void ExprOob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprOob*)self)->describe().c_ize() );
   
   ctx->topData().setOob();
}

void ExprOob::describeTo( String& str ) const
{
   str = "^+";
   str += m_first->describe();
}



bool ExprDeoob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.resetOob();
      return true;
   }

   return false;
}

void ExprDeoob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprDeoob*)self)->describe().c_ize() );
   
   ctx->topData().resetOob();
}

void ExprDeoob::describeTo( String& str ) const
{
   str = "^-";
   str += m_first->describe();
}



bool ExprXorOob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.xorOob();
      return true;
   }

   return false;
}

void ExprXorOob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprXorOob*)self)->describe().c_ize() );
   
   ctx->topData().xorOob();
}

void ExprXorOob::describeTo( String& str ) const
{
   str = "^%";
   str += m_first->describe();
}



bool ExprIsOob::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      value.setBoolean(value.isOob());
      return true;
   }

   return false;
}

void ExprIsOob::apply_( const PStep* DEBUG_ONLY(self), VMContext* ctx )
{  
   TRACE2( "Apply \"%s\"", ((ExprXorOob*)self)->describe().c_ize() );
   
   Item& item = ctx->topData();
   item.setBoolean(item.isOob());
}

void ExprIsOob::describeTo( String& str ) const
{
   str = "^?";
   str += m_first->describe();
}

//================================================
// Self
//

ExprSelf::ExprSelf():
   Expression(Expression::t_self)
{
   apply = apply_;
}

ExprSelf::ExprSelf( const ExprSelf &other ):
   Expression(other)
{
   apply = apply_;
}

ExprSelf::~ExprSelf() {}


bool ExprSelf::isStatic() const
{
   return false;
}

ExprSelf* ExprSelf::clone() const
{
   return new ExprSelf( *this );
}

bool ExprSelf::simplify( Item& ) const
{
   return false;
}

void ExprSelf::describeTo( String & str ) const
{
   str = "self";
}

void ExprSelf::apply_( const PStep*, VMContext* ctx )
{
   ctx->pushData(ctx->currentFrame().m_self);
}

}

/* end of expression.cpp */
