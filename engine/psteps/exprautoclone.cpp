/*
   FALCON - The Falcon Programming Language.
   FILE: exprautoclone.h

   Syntactic tree item definitions -- expression elements -- clone.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Apr 2013 16:04:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprautoclone.cpp"

#include <falcon/stream.h>
#include <falcon/item.h>
#include <falcon/gclock.h>
#include <falcon/vm.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/class.h>

#include <falcon/synclasses_id.h>

#include <falcon/psteps/exprautoclone.h>

namespace Falcon {

ExprAutoClone::ExprAutoClone( int line, int chr ):
   Expression(line, chr),
   m_cls(0),
   m_data(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_autoclone )
   apply = apply_;
}


ExprAutoClone::ExprAutoClone( const Class* cls, void* data, int line, int chr ):
      Expression( line, chr ),
      m_cls(cls),
      m_data(data)
{
   FALCON_DECLARE_SYN_CLASS( expr_autoclone )
   //m_handler = Engine::instance()->synclasses()->m_expr_autoclone;
   apply = apply_;
}

ExprAutoClone::ExprAutoClone( const ExprAutoClone& other ):
      Expression( other ),
      m_cls(other.m_cls),
      m_data(0)
{
   if ( m_cls != 0 )
   {
      m_data = m_cls->clone(other.m_data);
   }
}


ExprAutoClone::~ExprAutoClone()
{
   if ( m_cls != 0 )
   {
      m_cls->dispose( m_data );
      m_cls = 0;
      m_data = 0;
   }   
}


void ExprAutoClone::apply_( const PStep *ps, VMContext* ctx )
{
   const ExprAutoClone* self = static_cast<const ExprAutoClone*>(ps);
   ctx->popCode();
   void* ndata = self->m_cls->clone(self->m_data);
   ctx->pushData( FALCON_GC_STORE( self->m_cls, ndata ) );
}


void ExprAutoClone::set( Class* cls, void* data )
{
   if( m_cls != 0 )
   {
      m_cls->dispose(m_data);
   }
   m_cls = cls;
   m_data = data;
}


ExprAutoClone* ExprAutoClone::clone() const
{
   return new ExprAutoClone(*this);
}

void ExprAutoClone::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_cls != 0 )
   {
      String str;
      m_cls->describe( m_data, str, 1, -1 );
      tw->write( str );
   }
   else {
      tw->write("\"Empty ExprAutoClone\"");
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}


}

/* end of exprvalue.cpp */
