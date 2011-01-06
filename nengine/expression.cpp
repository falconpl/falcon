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
#include <falcon/error.h>
#include <falcon/item.h>

namespace Falcon {

Expression::Expression( const Expression &other ):
   m_operator( other.m_operator ),
   m_sourceRef( other.m_sourceRef )
{}

Expression::~Expression()
{}

void Expression::serialize( Stream* s )
{
   byte type = (byte) m_operator;
   s->write( &b, 1 );
   m_sourceRef.serialize( s );
}

void Expression::deserialize( Stream* s )
{
   m_sourceRef.deserialize( s );
}

void Expression::perform( VMachine* vm ) const
{
   Item dummy;
   evaluate( vm, dummy );
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
   delete m_second;
   delete m_third;
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

void TernaryExpression::serialize( Stream* s ) const
{
   Expression::serialize( s );
   m_first->serialize( s );
   m_second->serialize( s );
   m_third->serialize( s );
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
   return m_first->isStatic() && m_second->isStatic() && third->isStatic();
}

//=============================================================

Expression* ExprFactory::make( Expression::operator_t type )
{
   switch( type )
   {
   case t_value: return new ExprValue;
   case t_neg: return new ExprNeg;
   case t_not: return new ExprNot;

   case t_and: return new ExprAnd;
   case t_gate_and: return 0; // cannot be serialized
   case t_or: return new ExprOr;
   case t_gate_or: return 0; // cannot be serialized

   case t_plus: return new ExprPlus;
   case t_minus: return new ExprMinus;
   case t_times: return new ExprTimes;
   case t_divide: return new ExprDiv;
   case t_modulo: return new ExprMod;
   case t_power: return new ExprPow;

   case t_pre_inc: return new ExprPreInc;
   case t_post_inc: return new ExprPostInc;
   case t_pre_dec: return new ExprPreDec;
   case t_post_dec: return new ExprPostDec;

   case t_gt: return new ExprGT;
   case t_ge: return new ExprGE;
   case t_lt: return new ExprLT;
   case t_le: return new ExprLE;
   case t_eq: return new ExprEQ;
   case t_exeq: return new ExprEEQ;
   case t_neq: return new ExprNE;

   case t_in: return new ExprIn;
   case t_notin: return new ExprNotIn;
   case t_provides: return new ExprProvides;

   case t_iif: return new ExprIIF;

   case t_obj_access: return new ExprDot;
   case t_funcall: return new ExprCall;
   case t_array_access: return new ExprIndex;
   case t_array_byte_access: return new ExprStarIndex;
   case t_strexpand: return new ExprStrExpand;
   case t_indirect: return new ExprIndirect;

   case t_assign: return new ExprAssign;
   case t_fbind: return new ExprFbind;

   case t_aadd: return new ExprAutoAdd;
   case t_asub: return new ExprAutoSub;
   case t_amul: return new ExprAutoMul;
   case t_adiv: return new ExprAutoDiv;
   case t_amod: return new ExprAutoMod;
   case t_apow: return new ExprAutoPow;

   case t_eval: return new ExprEval;
   case t_oob: return new ExprOob;
   case t_deoob: return new ExprDeoob;
   case t_xoroob: return new ExprXorOob;
   case t_isoob: return new ExprIsOob;
   }
   return 0;
}


Expression* ExprFactory::deserialize( Stream* s )
{
   byte b;
   s->read( &b, 1 );
   operator_t type = reinterpret_cast<operator_t>( b );

   Expression* expr = make( type );
   if ( expr == 0 )
   {
      throw new IoError(ErrorParam( e_deser, __LINE__ ).extra( "Expression.deserialize"));
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

//==================================================================
// Expressions
//


void ExprNeg::evaluate( VMachine* vm, Item& value ) const
{
   if ( m_first->isDirect() )
   {
      Item operand;
      m_first->evaluate( vm, operand );
      Engine::getMetaClass( operand.type() )->neg( value, operand );
   }
   else {
      vm->pushCode( this );
      vm->pushCode( m_first );
   }
}

void ExprNeg::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   Engine::getMetaClass( operand.type() )->neg( operand, operand );
}

void ExprNeg::toString( String& str ) const
{
   str = "-";
   str += m_operand->toString();
}


// ===================== logic not.

void ExprNot::evaluate( VMachine* vm, Item& value ) const
{
   if ( m_first->isDirect() )
   {
      Item operand;
      m_first->evaluate( vm, operand );
      value.setBoolean( ! operand.isTrue() );
   }
   else {
      vm->pushCode( this );
      vm->pushCode( m_first );
   }
}

void ExprNot::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   operand.setBoolean( ! operand.isTrue() );
}

void ExprNot::toString( String& str ) const
{
   str = "not ";
   str += m_operand->toString();
}

//=========================================================

void ExprAnd::evaluate( VMachine* vm, Item& value ) const
{
   if ( m_first->isDirect() && m_second->isDirect() )
   {
      Item operand;
      m_first->evaluate( vm, operand );
      if( operand.isTrue() )
      {
         m_second->evaluate( vm, operand );
         value.setBoolean( operand.isTrue() );
      }
      else {
         value.setBoolean( false );
      }
   }
   else {
      vm->pushCode( this );
      vm->pushCode( m_second );
      vm->pushCode( &this->m_gate );
      vm->pushCode( m_first );
   }
}

void ExprAnd::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   operand.setBoolean( operand.isTrue() );
}

void ExprAnd::toString( String& str ) const
{
   str = "(" + m_first->toString() + " and " + m_second->toString() + ")";
}

void ExprAnd::Gate::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   if( operand.isFalse() )
   {
      operand.setBoolean( false );
      // we have already been popped; pop also m_first and the original and
      vm->popCode(2);
   }
   // otherwise just proceed.
}


//=========================================================

void ExprOr::evaluate( VMachine* vm, Item& value ) const
{
   if ( m_first->isDirect() && m_second->isDirect() )
   {
      Item operand;
      m_first->evaluate( vm, operand );
      if( operand.isFalse() )
      {
         m_second->evaluate( vm, operand );
         value.setBoolean( operand.isTrue() );
      }
      else {
         value.setBoolean( true );
      }
   }
   else {
      vm->pushCode( this );
      vm->pushCode( m_second );
      vm->pushCode( &this->m_gate );
      vm->pushCode( m_first );
   }
}

void ExprOr::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   operand.setBoolean( operand.isTrue() );
}

void ExprOr::toString( String& str ) const
{
   str = "(" + m_first->toString() + " or " + m_second->toString() + ")";
}

void ExprOr::Gate::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   if( operand.isTrue() )
   {
      operand.setBoolean( true );
      // we have already been popped; pop also m_first and the original and
      vm->popCode(2);
   }
   // otherwise just proceed.
}


//=========================================================

void ExprAssign::evaluate( VMachine* vm, Item& value ) const
{
   if ( m_first->isDirect() && m_second->isDirect() )
   {
      Item operand;
      m_second->evaluate( vm, operand );
      m_first->leval( vm, operand, value );
   }
   else {
      vm->pushCode( this );
      vm->pushCode( m_first );
      vm->pushCode( m_second );
   }
}

void ExprAssign::apply( VMachine* vm ) const
{
   Item& operand = vm->data(0);
   m_first->leval( vm, operand, operand );
}

void ExprAssign::toString( String& str ) const
{
   str = m_first->toString() + " = " + m_second->toString();
}


}

/* end of expression.cpp */
