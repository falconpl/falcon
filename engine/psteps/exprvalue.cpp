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

#undef SRC
#define SRC "engine/psteps/exprvalue.cpp"

#include <falcon/stream.h>
#include <falcon/item.h>
#include <falcon/gclock.h>
#include <falcon/vm.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/class.h>

#include <falcon/synclasses_id.h>

#include <falcon/psteps/exprvalue.h>

namespace Falcon {

ExprValue::ExprValue( int line, int chr ):
   Expression(line, chr),
   m_lock(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_value )
   apply = apply_;
   m_trait = e_trait_value;
}

ExprValue::ExprValue( const Item& item, int line, int chr ):
      Expression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_value )
   static Collector* coll = Engine::instance()->collector();

   m_item.copy(item); // silently copy
   apply = apply_;
   m_trait = e_trait_value;

   if ( item.isUser() )
   {
      m_lock = coll->lock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}


ExprValue::ExprValue( const ExprValue& other ):
   Expression( other ),
   m_item( other.m_item )
{
   static Collector* coll = Engine::instance()->collector();
   
   apply = apply_;
   m_trait = e_trait_value;

   if ( m_item.isUser() )
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
   if ( m_item.isUser() )
   {
      m_lock = coll->lock(m_item);
   }
   else
   {
      m_lock = 0;
   }
}


void ExprValue::apply_( const PStep *ps, VMContext* ctx )
{
   const ExprValue* self = static_cast<const ExprValue*>(ps);
   ctx->popCode();
   ctx->pushData( self->m_item );
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



void ExprValue::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   String str;
   Class* cls;
   void* inst;
   m_item.forceClassInst(cls, inst);

   // we'll need a special treatment for anon function and classes.
   if( cls->typeID() == FLC_CLASS_ID_FUNC || cls->typeID() == FLC_CLASS_ID_CLASS )
   {
      Mantra* mantra = static_cast<Mantra*>(inst);
      if( mantra->name() == "" || mantra->name().startsWith("_anon#") )
      {
         mantra->render( tw, depth );
      }
      else {
         str = mantra->name();
      }
   }
   else {
      cls->describe(inst, str, 1, -1 );
   }

   tw->write( str );

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}

bool ExprValue::isStandAlone() const
{
   if ( m_parent == 0 ) {
      return false;
   }

   return m_item.isBoolean() && m_parent->handler()->userFlags() == FALCON_SYNCLASS_ID_RULE;
}

}

/* end of exprvalue.cpp */
