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
#include <falcon/error.h>
#include <falcon/item.h>

namespace Falcon {

ExprValue::ExprValue( const Item& item ):
      Expression( t_value ),
      m_item(item)
{
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

void ExprValue::perform( VMachine* vm ) const
{
   vm->pushCode( this );
}


void ExprValue::apply( VMachine* vm ) const
{
   vm->pushData( m_item );
}


bool ExprValue::simplify( Item& item ) const
{
   item = m_item;
   return true;
}

virtual ExprValue* ExprValue::clone() const
{
   return new ExprValue( *this );
}

virtual void ExprValue::serialize( Stream* s ) const
{
   Expression::serialize( s );
   m_item.serialize(s);
}

virtual bool ExprValue::isStatic() const
{
   return true;
}

virtual const String ExprValue::toString() const
{
   return m_item.toString();
}

void ExprValue::deserialize( Stream* s )
{
   Expression::deserialize(s);
   m_item.deserialize();
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
