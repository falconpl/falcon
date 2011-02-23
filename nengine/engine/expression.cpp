/*
   FALCON - The Falcon Programming Language.
   FILE: expression.cpp

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/stream.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcode.h>
#include <falcon/exprfactory.h>
#include <falcon/operanderror.h>
#include <falcon/codeerror.h>
#include <math.h>

#include <falcon/trace.h>


namespace Falcon {

Expression::Expression( const Expression &other ):
   m_operator( other.m_operator ),
   m_sourceRef( other.m_sourceRef )
{}

Expression::~Expression()
{}

void Expression::serialize( Stream* s ) const
{
   byte type = (byte) m_operator;
   s->write( &type, 1 );
   m_sourceRef.serialize( s );
}

void Expression::deserialize( Stream* s )
{
   m_sourceRef.deserialize( s );
}

void Expression::precompile( PCode* pcode ) const
{
   pcode->pushStep( this );
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
   pcode->pushStep( this );
   m_first->precompile( pcode );
}

void UnaryExpression::serialize( Stream* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
}

void UnaryExpression::deserialize( Stream* s )
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
   pcode->pushStep( this );
   m_second->precompile( pcode );
   m_first->precompile( pcode );
}

void BinaryExpression::serialize( Stream* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
   m_second->serialize( s );
}

void BinaryExpression::deserialize( Stream* s )
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
   
   pcode->pushStep( this );
   m_third->precompile( pcode );
   m_second->precompile( pcode );
   m_first->precompile( pcode );
}


void TernaryExpression::serialize( Stream* s ) const
{
   Expression::serialize( s );
   m_third->serialize( s );
   m_second->serialize( s );
   m_first->serialize( s );
}

void TernaryExpression::deserialize( Stream* s )
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

void ExprNot::apply_( const PStep* self, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprNot*)self)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();
   Item& operand = ctx->topData();

   // remove ourselves
   ctx->popCode();

   //TODO: overload not
   operand.setBoolean( ! operand.isTrue() );
}

void ExprNot::describe( String& str ) const
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
   int shortCircuitSize = pcode->size();

   pcode->pushStep( this );
   // and then the second expr last
   m_second->precompile( pcode );
   // add a gate to jump checks on short circuits
   m_gate.m_shortCircuitSeqId = shortCircuitSize;
   pcode->pushStep( &m_gate );
   // check the first expression for first...
   m_first->precompile( pcode );
}

void ExprAnd::apply_( const PStep* self, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprAnd*)self)->describe().c_ize() );
   register VMContext* ctx = vm->currentContext();

   // use the space left from us by the previous expression
   Item& operand = ctx->topData();

   // Booleanize it
   operand.setBoolean( operand.isTrue() );
   // and remove ourselves
   ctx->popCode();

}

void ExprAnd::describe( String& str ) const
{
   str = "(" + m_first->describe() + " and " + m_second->describe() + ")";
}


ExprAnd::Gate::Gate() {
   apply = apply_;
}


void ExprAnd::Gate::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprAnd::Gate*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

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

   int shortCircuitSize = pcode->size();

   pcode->pushStep( this );
   // and then the second expr last
   m_second->precompile( pcode );
   // add a gate to jump checks on short circuits
   m_gate.m_shortCircuitSeqId = shortCircuitSize;
   pcode->pushStep( &m_gate );
   // check the first expression for first...
   m_first->precompile( pcode );
}


void ExprOr::apply_( const PStep* self, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprOr*)self)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();
   
   // reuse the operand left by the other expression
   Item& operand = ctx->topData();
   operand.setBoolean( operand.isTrue() );
   // remove ourselves
   ctx->popCode();

   
}

void ExprOr::describe( String& str ) const
{
   str = "(" + m_first->describe() + " or " + m_second->describe() + ")";
}

ExprOr::Gate::Gate() {
   apply = apply_;
}

void ExprOr::Gate::apply_( const PStep* ps,  VMachine* vm )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprOr::Gate*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

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
   m_first->precompile(pcode);
   m_second->precompile(pcode);
}


bool ExprAssign::simplify( Item& value ) const
{
   // TODO Simplify for closed symbols
   return false;
}

void ExprAssign::describe( String& str ) const
{
   str = m_first->describe() + " = " + m_second->describe();
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

void ExprNeg::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprNeg*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( -item.asInteger() ); break;
      case FLC_ITEM_NUM: item.setNumeric( -item.asNumeric() ); break;
      case FLC_ITEM_DEEP:
         item.asDeepClass()->op_neg( vm, item.asDeepInst(), item );
         break;

      case FLC_ITEM_USER:
         item.asUserClass()->op_neg( vm, item.asUserInst(), item );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("neg") ) );
   }
}

void ExprNeg::describe( String& str ) const
{
   str = "-";
   str += m_first->describe();
}



bool ExprPreInc::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( value.asInteger()+1 ); return true;
      case FLC_ITEM_NUM: value.setNumeric( value.asNumeric()+1 ); return true;
      }
   }

   return false;
}

void ExprPreInc::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprPreInc*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( item.asInteger()+1 ); break;
      case FLC_ITEM_NUM: item.setNumeric( item.asNumeric()+1 ); break;
      case FLC_ITEM_DEEP:
         item.asDeepClass()->op_inc( vm, item.asDeepInst(), item );
         break;

      case FLC_ITEM_USER:
         item.asUserClass()->op_inc( vm, item.asUserInst(), item );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("preinc") ) );
   }
}

void ExprPreInc::describe( String& str ) const
{
   str = "++";
   str += m_first->describe();
}


bool ExprPostInc::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( value.asInteger()+1 ); return true;
      case FLC_ITEM_NUM: value.setNumeric( value.asNumeric()+1 ); return true;
      }
   }

   return false;
}


void ExprPostInc::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );

   pcode->pushStep( this );
   pcode->pushStep( &m_gate );
   m_first->precompile( pcode );
}

void ExprPostInc::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprPostInc*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   
}

void ExprPostInc::describe( String& str ) const
{
   str = m_first->describe();
   str += "++";
}

ExprPostInc::Gate::Gate() {
   apply = apply_;
}

void ExprPostInc::Gate::apply_( const PStep* ps,  VMachine* vm )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprPostInc::Gate*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // read and recycle the topmost data.
   Item& operand = ctx->topData();

   switch( operand.type() )
   {
      case FLC_ITEM_INT: operand.setInteger( operand.asInteger()+1 ); break;
      case FLC_ITEM_NUM: operand.setNumeric( operand.asNumeric()+1 ); break;
      case FLC_ITEM_DEEP:
         operand.asDeepClass()->op_incpost( vm, operand.asDeepInst(), operand );
         break;

      case FLC_ITEM_USER:
         operand.asUserClass()->op_incpost( vm, operand.asUserInst(), operand );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("postinc") ) );
   }
   
}



bool ExprPreDec::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( value.asInteger()-1 ); return true;
      case FLC_ITEM_NUM: value.setNumeric( value.asNumeric()-1 ); return true;
      }
   }

   return false;
}

void ExprPreDec::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprPreDec*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( item.asInteger()-1 ); break;
      case FLC_ITEM_NUM: item.setNumeric( item.asNumeric()-1 ); break;
      case FLC_ITEM_DEEP:
         item.asDeepClass()->op_dec( vm, item.asDeepInst(), item );
         break;

      case FLC_ITEM_USER:
         item.asUserClass()->op_dec( vm, item.asUserInst(), item );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("predec") ) );
   }
}

void ExprPreDec::describe( String& str ) const
{
   str = "--";
   str += m_first->describe();
}



bool ExprPostDec::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( value.asInteger()-1 ); return true;
      case FLC_ITEM_NUM: value.setNumeric( value.asNumeric()-1 ); return true;
      }
   }

   return false;
}


void ExprPostDec::precompile( PCode* pcode ) const
{
   TRACE2( "Precompile \"%s\"", describe().c_ize() );

   pcode->pushStep( this );
   pcode->pushStep( &m_gate );
   m_first->precompile( pcode );
}

void ExprPostDec::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprPostDec*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   
}

void ExprPostDec::describe( String& str ) const
{
   str = m_first->describe();
   str += "--";
}

ExprPostDec::Gate::Gate() {
   apply = apply_;
}

void ExprPostDec::Gate::apply_( const PStep* ps,  VMachine* vm )
{
   TRACE2( "Apply GATE \"%s\"", ((ExprPostDec::Gate*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // read and recycle the topmost data.
   Item& operand = ctx->topData();

   switch( operand.type() )
   {
      case FLC_ITEM_INT: operand.setInteger( operand.asInteger()-1 ); break;
      case FLC_ITEM_NUM: operand.setNumeric( operand.asNumeric()-1 ); break;
      case FLC_ITEM_DEEP:
         operand.asDeepClass()->op_decpost( vm, operand.asDeepInst(), operand );
         break;

      case FLC_ITEM_USER:
         operand.asUserClass()->op_decpost( vm, operand.asUserInst(), operand );
         break;

      default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("postdec") ) );
   }
   
}

//=========================================================
//Binary

#define caseDeep \
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_NIL:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_BOOL:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_INT:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_NUM:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_METHOD:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_FUNC:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_BASEMETHOD:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_DEEP:\
      case FLC_ITEM_DEEP << 8 | FLC_ITEM_USER

#define caseUser \
      case FLC_ITEM_USER << 8 | FLC_ITEM_NIL:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_BOOL:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_INT:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_NUM:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_METHOD:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_FUNC:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_BASEMETHOD:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_DEEP:\
      case FLC_ITEM_USER << 8 | FLC_ITEM_USER

bool ExprPlus::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( d1.asInteger() + d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asInteger() + d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( d1.asNumeric() + d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asNumeric() + d2.asNumeric() );
         return true;
      }
   }

   return false;
}


void ExprPlus::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprPlus*)ps)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();

   // No need to copy the second, we're not packing the stack now.
   Item& d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setInteger( d1.asInteger() + d2.asInteger() );
      break;
   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asInteger() + d2.asNumeric() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setNumeric( d1.asNumeric() + d2.asInteger() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asNumeric() + d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_add( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_add( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("+") ) );
   }


}


void ExprPlus::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " + " + m_second->describe() + ")";
}


bool ExprMinus::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( d1.asInteger() - d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asInteger() - d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( d1.asNumeric() - d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asNumeric() - d2.asNumeric() );
         return true;
      }
   }

   return false;
}


void ExprMinus::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprMinus*)ps)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();

   // No need to copy the second, we're not packing the stack now.
   Item& d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setInteger( d1.asInteger() - d2.asInteger() );
      break;
   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asInteger() - d2.asNumeric() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setNumeric( d1.asNumeric() - d2.asInteger() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asNumeric() - d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_sub( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_sub( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("-") ) );
   }


}


void ExprMinus::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " - " + m_second->describe() + ")";
}



bool ExprTimes::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( d1.asInteger() * d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asInteger() * d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( d1.asNumeric() * d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asNumeric() * d2.asNumeric() );
         return true;
      }
   }

   return false;
}


void ExprTimes::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprTimes*)ps)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();

   // No need to copy the second, we're not packing the stack now.
   Item& d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setInteger( d1.asInteger() * d2.asInteger() );
      break;
   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asInteger() * d2.asNumeric() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setNumeric( d1.asNumeric() * d2.asInteger() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asNumeric() * d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_mul( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_mul( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("*") ) );
   }


}


void ExprTimes::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " * " + m_second->describe() + ")";
}



bool ExprDiv::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
   {
      if ( d2.isOrdinal() && d2.forceInteger() == 0 )
      {
         throw new CodeError( ErrorParam(e_div_by_zero, __LINE__).origin(ErrorParam::e_orig_compiler) );
      }
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( d1.asInteger() / d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asInteger() / d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( d1.asNumeric() / d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( d1.asNumeric() / d2.asNumeric() );
         return true;
      }
   }

   return false;
}


void ExprDiv::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprDiv*)ps)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();

   // No need to copy the second, we're not packing the stack now.
   Item& d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   if ( d2.isOrdinal() && d2.forceInteger() == 0 )
   {
      throw new CodeError( ErrorParam(e_div_by_zero, __LINE__).origin(ErrorParam::e_orig_vm) );
   }

   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setInteger( d1.asInteger() / d2.asInteger() );
      break;
   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asInteger() / d2.asNumeric() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setNumeric( d1.asNumeric() / d2.asInteger() );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setNumeric( d1.asNumeric() / d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_div( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_div( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("/") ) );
   }


}


void ExprDiv::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " / " + m_second->describe() + ")";
}



bool ExprMod::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
   {
      if ( d2.isOrdinal() && d2.forceInteger() == 0 )
      {
         throw new CodeError( ErrorParam(e_mod_by_zero, __LINE__).origin(ErrorParam::e_orig_compiler) );
      }
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setInteger( d1.asInteger() % d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setNumeric( fmod( d1.asInteger(), d2.asNumeric() ) );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setNumeric( fmod( d1.asNumeric(), d2.asInteger() ) );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( fmod( d1.asNumeric(), d2.asNumeric() ) );
         return true;
      }
   }

   return false;
}


void ExprMod::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprMod*)ps)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();

   // No need to copy the second, we're not packing the stack now.
   Item& d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   if ( d2.isOrdinal() && d2.forceInteger() == 0 )
   {
      throw new CodeError( ErrorParam(e_mod_by_zero, __LINE__).origin(ErrorParam::e_orig_vm) );
   }

   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setInteger( d1.asInteger() % d2.asInteger() );
      break;
   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setNumeric( fmod( d1.asInteger(), d2.asNumeric() ) );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setNumeric( fmod( d1.asNumeric(), d2.asInteger() ) );
      break;
   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setNumeric( fmod( d1.asNumeric(), d2.asNumeric() ) );
      break;

   caseDeep:
      d1.asDeepClass()->op_mod( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_mod( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("%") ) );
   }


}


void ExprMod::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " % " + m_second->describe() + ")";
}



bool ExprPow::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setNumeric( pow( d1.forceNumeric(), d2.forceNumeric() ) );
         return true;
      }
   }

   return false;
}


void ExprPow::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprPow*)ps)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();

   // No need to copy the second, we're not packing the stack now.
   Item& d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   switch ( d1.type() << 8 | d2.type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         d1.setNumeric( pow( d1.forceNumeric(), d2.forceNumeric() ) );
         break;

   caseDeep:
      d1.asDeepClass()->op_pow( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_pow( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("**") ) );
   }


}


void ExprPow::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " ** " + m_second->describe() + ")";
}

//=========================================================
//Comparisons


bool ExprLT::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() < d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() < d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() < d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() < d2.asNumeric() );
         return true;
      default:
         if ( d1.type() < d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprLT::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprLT*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() < d2.asInteger() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asInteger() < d2.asNumeric() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asNumeric() < d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() < d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_lt( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_lt( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("<") ) );
   }
}


void ExprLT::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " < " + m_second->describe() + ")";
}



bool ExprLE::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() <= d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() <= d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() <= d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() <= d2.asNumeric() );
         return true;
      default:
         if ( d1.type() <= d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprLE::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprLE*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() <= d2.asInteger() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asInteger() <= d2.asNumeric() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asNumeric() <= d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() <= d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_le( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_le( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("<=") ) );
   }
}


void ExprLE::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " <= " + m_second->describe() + ")";
}



bool ExprGT::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() > d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() > d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() > d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() > d2.asNumeric() );
         return true;
      default:
         if ( d1.type() > d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprGT::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprGT*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() > d2.asInteger() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asInteger() > d2.asNumeric() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asNumeric() > d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() > d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_gt( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_gt( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra(">") ) );
   }
}


void ExprGT::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " > " + m_second->describe() + ")";
}



bool ExprGE::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() >= d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() >= d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() >= d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() >= d2.asNumeric() );
         return true;
      default:
         if ( d1.type() >= d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprGE::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprGE*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() >= d2.asInteger() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asInteger() >= d2.asNumeric() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asNumeric() >= d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() >= d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_ge( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_ge( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra(">=") ) );
   }
}


void ExprGE::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " >= " + m_second->describe() + ")";
}




bool ExprEQ::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() == d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() == d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() == d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() == d2.asNumeric() );
         return true;
      default:
         if ( d1.type() == d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprEQ::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprEQ*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() == d2.asInteger() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asInteger() == d2.asNumeric() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asNumeric() == d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() == d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_eq( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_eq( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("==") ) );
   }
}


void ExprEQ::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " == " + m_second->describe() + ")";
}



bool ExprEEQ::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() == d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() == d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() == d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() == d2.asNumeric() );
         return true;
      default:
         if ( d1.type() == d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprEEQ::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprEEQ*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() == d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() == d2.asNumeric() );
      break;

   case FLC_ITEM_DEEP << 8 | FLC_ITEM_DEEP:
      d1.setBoolean( d1.asDeepInst() == d2.asDeepInst() );
      break;

   case FLC_ITEM_USER << 8 | FLC_ITEM_USER:
      d1.setBoolean( d1.asUserInst() == d2.asUserInst() );
      break;

   default:
      d1.setBoolean(false);
   }
}


void ExprEEQ::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " === " + m_second->describe() + ")";
}



bool ExprNE::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_second->simplify(d2) )
   {
      switch ( d1.type() << 8 | d2.type() )
      {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asInteger() != d2.asInteger() );
         return true;
      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asInteger() != d2.asNumeric() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
         value.setBoolean( d1.asNumeric() != d2.asInteger() );
         return true;
      case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
         value.setBoolean( d1.asNumeric() != d2.asNumeric() );
         return true;
      default:
         if ( d1.type() != d2.type() )
         {
            value.setBoolean( true );
         }
         else
         {
            value.setBoolean( false );
         }
      }
   }

   return false;
}


void ExprNE::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprNE*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();


   switch ( d1.type() << 8 | d2.type() )
   {
   case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asInteger() != d2.asInteger() );
      break;

   case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asInteger() != d2.asNumeric() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_INT:
      d1.setBoolean( d1.asNumeric() != d2.asInteger() );
      break;

   case FLC_ITEM_NUM << 8 | FLC_ITEM_NUM:
      d1.setBoolean( d1.asNumeric() != d2.asNumeric() );
      break;

   caseDeep:
      d1.asDeepClass()->op_ne( vm, d1.asDeepInst(), d2, d1 );
      break;

   caseUser:
      d1.asUserClass()->op_ne( vm, d1.asUserInst(), d2, d1 );
      break;

   default:
      // no need to throw, we're going to get back in the VM.
      vm->raiseError(
         new OperandError( ErrorParam(e_invalid_op, __LINE__ ).extra("!=") ) );
   }
}


void ExprNE::describe( String& ret ) const
{
   ret = "(" + m_first->describe() + " != " + m_second->describe() + ")";
}


//=========================================================
// Call


ExprCall::~ExprCall()
{
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < m_params.size(); ++i )
   {
      delete m_params[i];
   }
}


/** Function call. */
void ExprCall::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling call: %p (%s)", pcode, describe().c_ize() );

   // set our callback
   pcode->pushStep( this );
   m_first->precompile( pcode );

   // and generate all the expressions, in inverse order.
   for( int i = (int) m_params.size()-1; i >= 0; --i )
   {
      m_params[i]->precompile( pcode );
   }
}


bool ExprCall::simplify( Item& value ) const
{
   return false;
}

void ExprCall::apply_( const PStep* v, VMachine* vm )
{
   TRACE2( "Apply CALL", 1 );
   
   const ExprCall* self = static_cast<const ExprCall*>(v);
   register VMContext* ctx = vm->currentContext();

   Function* f = ctx->topData().asFunction();
   ctx->popData();
   vm->call( f, self->paramCount() );
}


ExprCall& ExprCall::addParameter( Expression* p )
{
   m_params.push_back( p );
   return *this;
}


Expression* ExprCall::getParam( int n ) const
{
   return m_params[ n ];
}


void ExprCall::describe( String& ret ) const
{
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < m_params.size(); ++i )
   {
      if ( params.size() )
      {
         params += ", ";
      }
      params += m_params[i]->describe();
   }

   ret = m_first->describe() + "(" + params +  ")";
}

//=========================================================
//Accessors

bool ExprDot::simplify( Item& value ) const
{
   return false;
}


void ExprDot::apply_( const PStep* ps, VMachine* vm )
{
   TRACE2( "Apply \"%s\"", ((ExprDot*)ps)->describe().c_ize() );

   register VMContext* ctx = vm->currentContext();

   // copy the prop name
   Item prop = ctx->topData();
   ctx->popData();
   Class* cls;
   void* self;
   //acquire the class
   ctx->topData().forceClassInst(cls, self);
   if ( isLValue() )
   {
      ctx->popData();
      Item target = ctx->topData();
      cls->op_setProperty(vm, self, *prop.asString(), target);
   }
   else
   {
      cls->op_getProperty(vm, self, *prop.asString(), ctx->topData());
   }
}


void ExprDot::describe( String& ret ) const
{
   ret = "(" m_first->describe() + "." + m_second->describe() + ")";
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

void ExprOob::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprOob*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   ctx->topData().setOob();
   // remove ourselves
   ctx->popCode();
}

void ExprOob::describe( String& str ) const
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

void ExprDeoob::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprDeoob*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   ctx->topData().resetOob();
   // remove ourselves
   ctx->popCode();
}

void ExprDeoob::describe( String& str ) const
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

void ExprXorOob::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprXorOob*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   ctx->topData().xorOob();
   // remove ourselves
   ctx->popCode();
}

void ExprXorOob::describe( String& str ) const
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

void ExprIsOob::apply_( const PStep* self, VMachine* vm )
{  
   TRACE2( "Apply \"%s\"", ((ExprXorOob*)self)->describe().c_ize() );
   
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   item.setBoolean(item.isOob());
}

void ExprIsOob::describe( String& str ) const
{
   str = "^?";
   str += m_first->describe();
}


}

/* end of expression.cpp */
