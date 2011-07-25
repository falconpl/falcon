/*
   FALCON - The Falcon Programming Language.
   FILE: exprvalue.cpp

   Syntactic tree item definitions -- expression elements -- value.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/exprvalue.h>
#include <falcon/stream.h>
#include <falcon/item.h>
#include <falcon/gclock.h>
#include <falcon/vm.h>
#include <falcon/pcode.h>

namespace Falcon {

ExprValue::ExprValue( const Item& item ):
      Expression( t_value )
{
   static Collector* coll = Engine::instance()->collector();

   m_item.copy(item); // silently copy
   apply = apply_;

   if ( item.isGarbaged() )
   {
      m_lock = coll->lock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}

ExprValue::ExprValue( const ExprValue& other ):
   Expression( t_value ),
   m_item( other.m_item )
{
   static Collector* coll = Engine::instance()->collector();
   
   apply = apply_;

   if ( m_item.isGarbaged() )
   {
      m_lock = coll->lock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}


ExprValue::~ExprValue()
{
   if ( m_lock != 0 )
   {
      m_lock->dispose();
   }   
}


void ExprValue::item( const Item& i )
{
   static Collector* coll = Engine::instance()->collector();
   
   if ( m_lock != 0 )
   {
      m_lock->dispose();
   }   

   m_item = i;
   if ( m_item.isGarbaged() )
   {
      m_lock = coll->lock(m_item);
   }   
}


void ExprValue::apply_( const PStep *ps, VMContext* ctx )
{
   const ExprValue* self = static_cast<const ExprValue*>(ps);
   ctx->pushData( self->m_item );
}

void ExprValue::precompile( PCode* pc )  const
{
   pc->pushStep( this );
}

bool ExprValue::simplify( Item& item ) const
{
   item = m_item;
   return true;
}

ExprValue* ExprValue::clone() const
{
   return new ExprValue( *this );
}

void ExprValue::serialize( DataWriter* s ) const
{
   Expression::serialize( s );
   //m_item.serialize(s);
}

bool ExprValue::isStatic() const
{
   return true;
}

bool ExprValue::isBinaryOperator() const
{
   return false;
}

void ExprValue::describe( String & str ) const
{
   m_item.describe(str);
}

void ExprValue::deserialize( DataReader* s )
{
   static Collector* coll = Engine::instance()->collector();

   Expression::deserialize(s);
   //m_item.deserialize();
   if ( m_item.isGarbaged() )
   {
      m_lock = coll->lock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}

}

/* end of exprvalue.cpp */
