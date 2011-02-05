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
#include <falcon/garbagelock.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/pcode.h>

namespace Falcon {

ExprValue::ExprValue( const Item& item ):
      Expression( t_value ),
      m_item(item)
{
   apply = apply_;

   if ( item.isDeep() )
   {
      m_lock = new GarbageLock(item);
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
   apply = apply_;

   if ( m_item.isDeep() )
   {
     m_lock = new GarbageLock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}

ExprValue::~ExprValue()
{
   delete m_lock;
}

void ExprValue::item( const Item& i )
{
   m_item  = i;
   delete m_lock;
   if ( m_item.isDeep() )
   {
     m_lock = new GarbageLock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}


void ExprValue::apply_( const PStep *ps, VMachine* vm )
{
   const ExprValue* self = static_cast<const ExprValue*>(ps);

   vm->currentContext()->pushData( self->m_item );
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

void ExprValue::serialize( Stream* s ) const
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

void ExprValue::toString( String & str ) const
{
   m_item.toString(str);
}

void ExprValue::deserialize( Stream* s )
{
   Expression::deserialize(s);
   //m_item.deserialize();
   if ( m_item.isDeep() )
   {
     m_lock = new GarbageLock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}

}

/* end of exprvalue.cpp */
