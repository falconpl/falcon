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
   TRACE3( "Precompiling un-exp: %p (%s)", pcode, toString().c_ize() );
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
   TRACE3( "Precompiling bin-exp: %p (%s)", pcode, toString().c_ize() );
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
   TRACE3( "Precompiling tri-exp: %p (%s)", pcode, toString().c_ize() );
   
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

bool ExprNeg::simplify( Item& value ) const
{
   if( m_first->simplify( value ) )
   {
      switch( value.type() )
      {
      case FLC_ITEM_INT: value.setInteger( -value.asInteger() ); return true;
      case FLC_ITEM_NUM: value.setNumeric( -value.asNumeric() ); return true;
      // TODO throw an exception, even if we shouldn't be here thanks to the compiler.
      default: return false;
      }
   }

   return false;
}

void ExprNeg::apply_( const PStep*, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();
   Item& item = ctx->topData();
   // remove ourselves
   ctx->popCode();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( -item.asInteger() ); break;
      case FLC_ITEM_NUM: item.setNumeric( -item.asNumeric() ); break;
      // TODO throw an exception, even if we shouldn't be here thanks to the compiler.
      case FLC_ITEM_NIL: case FLC_ITEM_BOOL: break;
      /*
      default:
         // TODO

         CoreObject* obj = item.asObject();
         obj->getClass()->__neg(item);
         */
   }
   
   TRACE2( "Apply NEG %d", (int) item.asInteger() );
}

void ExprNeg::toString( String& str ) const
{
   str = "-";
   str += m_first->toString();
}

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

void ExprNot::apply_( const PStep*, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();
   Item& operand = ctx->topData();

   // remove ourselves
   ctx->popCode();

   operand.setBoolean( ! operand.isTrue() );
   TRACE2( "Apply NOT %s", operand.isTrue() ? "true" : "false" );
}

void ExprNot::toString( String& str ) const
{
   str = "not ";
   str += m_first->toString();
}

//=========================================================
//


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
   int shortCircuitSize = pcode->size();

   TRACE3( "Precompiling and: %p (%s)", pcode, toString().c_ize() );

   pcode->pushStep( this );
   // and then the second expr last
   m_second->precompile( pcode );
   // add a gate to jump checks on short circuits
   m_gate.m_shortCircuitSeqId = shortCircuitSize;
   pcode->pushStep( &m_gate );
   // check the first expression for first...
   m_first->precompile( pcode );
}

void ExprAnd::apply_( const PStep*, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();

   // use the space left from us by the previous expression
   Item& operand = ctx->topData();
   // Booleanize it
   operand.setBoolean( operand.isTrue() );
   // and remove ourselves
   ctx->popCode();

   TRACE2( "Apply AND %s", operand.isTrue() ? "true" : "false" );
}

void ExprAnd::toString( String& str ) const
{
   str = "(" + m_first->toString() + " and " + m_second->toString() + ")";
}


ExprAnd::Gate::Gate() {
   apply = apply_;
}


void ExprAnd::Gate::apply_( const PStep* ps, VMachine* vm )
{
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

   TRACE2( "Apply AND::GATE %s", operand.isTrue() ? "true" : "false" );
}

//=========================================================

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
   int shortCircuitSize = pcode->size();

   TRACE3( "Precompiling or: %p (%s)", pcode, toString().c_ize() );
   
   pcode->pushStep( this );
   // and then the second expr last
   m_second->precompile( pcode );
   // add a gate to jump checks on short circuits
   m_gate.m_shortCircuitSeqId = shortCircuitSize;
   pcode->pushStep( &m_gate );
   // check the first expression for first...
   m_first->precompile( pcode );
}


void ExprOr::apply_( const PStep*, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();

   // reuse the operand left by the other expression
   Item& operand = ctx->topData();
   operand.setBoolean( operand.isTrue() );
   // remove ourselves
   ctx->popCode();

   TRACE2( "Apply OR %s", operand.isTrue() ? "true" : "false" );
}

void ExprOr::toString( String& str ) const
{
   str = "(" + m_first->toString() + " or " + m_second->toString() + ")";
}

ExprOr::Gate::Gate() {
   apply = apply_;
}

void ExprOr::Gate::apply_( const PStep* ps,  VMachine* vm )
{
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

   TRACE2( "Apply OR::GATE %s", operand.isTrue() ? "true" : "false" );
}

//=========================================================
//

void ExprAssign::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling Assign: %p (%s)", pcode, toString().c_ize() );

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

void ExprAssign::toString( String& str ) const
{
   str = m_first->toString() + " = " + m_second->toString();
}

//=========================================================
//

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


void ExprPlus::apply_( const PStep*, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();

   // copy the second
   Item d2 = ctx->topData();
   ctx->popData();
   // apply on the first
   Item& d1 = ctx->topData();

   TRACE2( "Apply PLUS", 1 );

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
   }

}


void ExprPlus::toString( String& ret ) const
{
   ret = m_first->toString() + " + " + m_second->toString();
}


//=========================================================
//


bool ExprLT::simplify( Item& value ) const
{
   Item d1, d2;
   if( m_first->simplify(d1) && m_first->simplify(d2) )
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
         value.setBoolean( d1.asNumeric()< d2.asNumeric() );
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


void ExprLT::apply_( const PStep*, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();

   TRACE2( "Apply LT", 1 );

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
      d1.setBoolean( d1.asNumeric()< d2.asNumeric() );
      break;
   default:
      if ( d1.type() < d2.type() )
      {
         d1.setBoolean( true );
      }
      else
      {
         d1.setBoolean( false );
      }
   }
}


void ExprLT::toString( String& ret ) const
{
   ret = m_first->toString() + " < " + m_second->toString();
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
   TRACE3( "Precompiling call: %p (%s)", pcode, toString().c_ize() );

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


void ExprCall::toString( String& ret ) const
{
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < m_params.size(); ++i )
   {
      if ( params.size() )
      {
         params += ", ";
      }
      params += m_params[i]->toString();
   }

   ret = m_first->toString() + "(" + params +  ")";
}


}

/* end of expression.cpp */
