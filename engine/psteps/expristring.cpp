/*
   FALCON - The Falcon Programming Language.
   FILE: expristring.cpp

   Syntactic tree item definitions -- expression elements -- i-string
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Mar 2013 20:23:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/expristring.cpp"

#include <falcon/stream.h>
#include <falcon/item.h>
#include <falcon/gclock.h>
#include <falcon/vm.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/class.h>

#include <falcon/synclasses_id.h>

#include <falcon/psteps/expristring.h>

namespace Falcon {

ExprIString::ExprIString( int line, int chr ):
   Expression(line, chr),
   m_tlgen(0),
   m_lock(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_istring )
   apply = apply_;

   m_lock = 0;
   m_resolved = 0;
}

ExprIString::ExprIString( const String& orig, int line, int chr ):
      Expression( line, chr ),
   m_tlgen(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_istring )
   apply = apply_;

   m_original = orig;
   m_lock = 0;
   m_resolved = 0;
}


ExprIString::ExprIString( const ExprIString& other ):
   Expression( other ),
   m_tlgen(0),
   m_original( m_original )
{
   m_lock = 0;
   m_resolved = 0;
}


ExprIString::~ExprIString()
{
   if ( m_lock != 0 )
   {
      m_lock->dispose();
   }
   // don't delete the resolved string.
}



void ExprIString::apply_( const PStep *ps, VMContext* ctx )
{
   const ExprIString* self = static_cast<const ExprIString*>(ps);
   ctx->popCode();
   String* res = self->resolve(ctx);
   ctx->pushData( Item(res->handler(),res) );
}

bool ExprIString::simplify( Item& ) const
{
   return false;
}

ExprIString* ExprIString::clone() const
{
   return new ExprIString( *this );
}


void ExprIString::render( TextWriter* tw, int depth ) const
{
   tw->write(renderPrefix(depth));

   String tgt;
   m_original.escape(tgt);

   tw->write("i\"");
   tw->write( tgt );
   tw->write("\"");

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


bool ExprIString::isStandAlone() const
{
   return false;
}


String* ExprIString::resolve( VMContext* ctx ) const
{
   Process* prc = ctx->process();
   String temp;

   if( prc->getTranslation(m_original, temp, m_tlgen ) )
   {
      // changed!
      String* changed =  new String( temp );
      GCLock* lock = FALCON_GC_STORELOCKED_SRCLINE(changed->handler(), changed, SRC, __LINE__ );
      m_mtx.lock();
      if( m_lock != 0 )
      {
         m_lock->dispose();
      }
      m_lock = lock;
      m_resolved = changed;
      m_mtx.unlock();

      return m_resolved;
   }
   else
   {
      m_mtx.lock();
      String* res = m_resolved;
      m_mtx.unlock();

      return res;
   }
}


void ExprIString::original( const String& orig )
{
   m_original = orig;
   m_tlgen = 0;
}

const String& ExprIString::original() const
{
   return m_original;
}

}

/* end of expristring.cpp */
