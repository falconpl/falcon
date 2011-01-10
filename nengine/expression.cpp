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
}

void UnaryExpression::perform( VMachine* vm ) const
{
   vm->pushCode( this );
   m_first->perform();
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


void BinaryExpression::perform( VMachine* vm ) const
{
   vm->pushCode( this );
   m_first->perform();
   m_second->perform();
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


void TernaryExpression::perform( VMachine* vm ) const
{
   // actually, ternary expressions doesn't require this fallback.
   vm->pushCode( this );
   m_first->perform();
   m_second->perform();
   m_third->perform();
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


//==================================================================
// Expressions
//

void ExprNeg::simplify( Item& value ) const
{
   if( m_first->simplify( item ) )
   {
      switch( item.type() )
      {
      case FLC_ITEM_INT: item.setInteger( -item.getInteger() ); return true;
      case FLC_ITEM_NUM: item.setNumeric( -item.getNumeric() ); return true;
      // TODO throw an exception, even if we shouldn't be here thanks to the compiler.
      default: return false;
      }
   }

   return false;
}

void ExprNeg::apply( VMachine* vm ) const
{
   Item& item = vm->topData();
   // remove ourselves
   vm->popCode();

   switch( item.type() )
   {
      case FLC_ITEM_INT: item.setInteger( -item.getInteger() ); break;
      case FLC_ITEM_NUM: item.setNumeric( -item.getNumeric() ); break;
      // TODO throw an exception, even if we shouldn't be here thanks to the compiler.
      case FLC_ITEM_NIL: case FLC_ITEM_BOOL: break;
      default:
         CoreObject* obj = item.asObject();
         obj->getClass()->__neg(item);
   }
}

void ExprNeg::toString( String& str ) const
{
   str = "-";
   str += m_operand->toString();
}

// ===================== logic not.

void ExprNot::simplify( Item& value ) const
{
   if( m_first->simplify( item ) )
   {
      item.setBoolean( ! item.isTrue() );
      return true;
   }
   return false;
}

void ExprNot::apply( VMachine* vm ) const
{
   Item& operand = vm->topData();

   // remove ourselves
   vm->popCode();

   operand.setBoolean( ! operand.isTrue() );
}

void ExprNot::toString( String& str ) const
{
   str = "not ";
   str += m_operand->toString();
}

//=========================================================

void ExprAnd::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      item.setBoolean( fi.isTrue() && si.isTrue() );
      return true;
   }
   return false;
}

void ExprAnd::perform( VMachine* vm ) const
{
   vm->pushCode( this );
   // add a gate...
   vm->pushCode( &m_gate );
   // check the first expression...
   m_first->perform();
}

void ExprAnd::apply( VMachine* vm ) const
{
   // use the space left from us by the previous expression
   Item& operand = vm->topData();
   // Booleanize it
   operand.setBoolean( operand.isTrue() );
   // and remove ourselves
   vm->popCode();
}

void ExprAnd::toString( String& str ) const
{
   str = "(" + m_first->toString() + " and " + m_second->toString() + ")";
}

void ExprAnd::Gate::apply( VMachine* vm ) const
{
   Item& operand = vm->topData();
   if( operand.isFalse() )
   {
      operand.setBoolean( false );
      // Pop ourselves and the calling expression.
      vm->popCode(2);
   }
   else {
      // just pop ourselves
      vm->popCode();
      // remove the data, which will be pushed by the other expression.
      vm->popData();

      // and proceed checking the other data.
      m_second->perform();
   }

}


//=========================================================

void ExprAnd::simplify( Item& value ) const
{
   Item fi, si;

   if( m_first->simplify( fi ) && m_second->simplify( si ) )
   {
      item.setBoolean( fi.isTrue() || si.isTrue() );
      return true;
   }
   return false;
}


void ExprOr::perform( VMachine* vm ) const
{
   vm->pushCode( this );
   // add a gate...
   vm->pushCode( &m_gate );

   // check the first expression
   m_first->perform();
}


void ExprOr::apply( VMachine* vm ) const
{
   // reuse the operand left by the other expression
   Item& operand = vm->topData();
   operand.setBoolean( operand.isTrue() );
   // remove ourselves
   vm->popCode();
}

void ExprOr::toString( String& str ) const
{
   str = "(" + m_first->toString() + " or " + m_second->toString() + ")";
}

void ExprOr::Gate::apply( VMachine* vm ) const
{
   // read and recycle the topmost data.
   Item& operand = vm->topData();
   if( operand.isTrue() )
   {
      operand.setBoolean( true );
      // pop ourselves and the calling code
      vm->popCode(2);
   }
   else {
      // pop ourselves
      vm->popCode();
      // as other expression is bound to push data, remove ours
      vm->popData();

      // and proceed checking the other expression.
      m_second->perform();
   }
}


//=========================================================

void ExprAssign::perform( VMachine* vm) const
{
   vm->pushCode( this );
   // First generate second assignand.
   // if that throws, there isn't any need to check for the assignment.
   m_second->perform();
}

void ExprAssign::apply( VMachine* vm ) const
{
   // delete ourselves
   vm->popCode();
   int size = vm->codeSize();

   // prepare the code for the assignee
   m_first->perform();

   // Note: one-level operands, as plain symbols, are just pushed in the code.
   // -- Deep assignments are left-to-right, meaning that the assignee l-value
   // -- is in the topmost expression. The topmost expression is the one that
   // -- is placed right where we stood (at 'size' position).
}

void ExprAssign::toString( String& str ) const
{
   str = m_first->toString() + " = " + m_second->toString();
}

}

/* end of expression.cpp */
