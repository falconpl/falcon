/*
   FALCON - The Falcon Programming Language.
   FILE: gc.cpp

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/gc.cpp"

#include <falcon/cm/gc.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {
namespace Ext {


ClassGC::ClassGC():
   ClassUser("%GC"),
   FALCON_INIT_PROPERTY( memory ),
   FALCON_INIT_PROPERTY( items ),
   FALCON_INIT_PROPERTY( enabled ),
   FALCON_INIT_PROPERTY( status ),

   FALCON_INIT_METHOD( perform ),
   FALCON_INIT_METHOD( suggest )
{
   // we don't need an object
   m_bIsFlatInstance = true;
}

ClassGC::~ClassGC()
{}


void* ClassGC::createInstance() const
{
   return 0;
}

void ClassGC::dispose( void* ) const
{
   // nothing to do
}

void* ClassGC::clone( void* instance ) const
{
   return instance;
}


void ClassGC::gcMarkInstance( void*, uint32 ) const
{
   // nothing to do
}

bool ClassGC::gcCheckInstance( void*, uint32 ) const
{
   // nothing to do
   return true;
}

bool ClassGC::op_init( VMContext* , void*, int  ) const
{
   // nothing to do
   return true;
}

//====================================================
// Properties.
//
   
FALCON_DEFINE_PROPERTY_SET_P0( ClassGC, memory )
{
   throw new CodeError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("memory") );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, memory )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->storedMemory();
}

FALCON_DEFINE_PROPERTY_SET_P0( ClassGC, items )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("items") );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, items )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->storedItems();
}

FALCON_DEFINE_PROPERTY_SET( ClassGC, enabled )(void*, const Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   coll->enable( value.isTrue() );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, enabled )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->isEnabled();
}

FALCON_DEFINE_PROPERTY_SET( ClassGC, status )(void*, const Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   int64 v = value.forceInteger();
   if( v < 0 || v > static_cast<int64>(Collector::e_status_red) )
   {
      throw FALCON_SIGN_ERROR( ParamError, e_param_range );
   }
   coll->status( static_cast<Collector::t_status>(v) );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, status )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = static_cast<int64>(coll->status());
}


FALCON_DEFINE_METHOD_P1( ClassGC, perform )
{
   static Collector* coll = Engine::instance()->collector();

   Item* i_full = ctx->param(0);
   Item* i_wait = ctx->param(1);

   bool full = i_full != 0 ? i_full->isTrue() : false;
   bool wait = i_wait != 0 ? i_wait->isTrue() : false;

   TRACE( "ClassGC::Method_perform %s, %s", (full? "Full" : "partial"), (wait?"wait": "no wait") );

   if( full )
   {
      if( wait != 0 )
      {
         Shared* sh = new Shared;
         coll->performGCOnShared( sh );
         ctx->addWait(sh);
         ctx->engageWait(-1);
      }
      else
      {
         coll->performGC(false);
      }
   }
   else
   {
      ctx->setInspectEvent();
   }

   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassGC, suggest )
{
   static Collector* coll = Engine::instance()->collector();

   Item* i_all = ctx->param(0);

   bool all = i_all != 0 ? i_all->isTrue() : false;

   TRACE( "ClassGC::Method_suggest %s", (all? "Full" : "partial") );

   coll->suggestGC(all);
   ctx->returnFrame();
}

}
}

/* end of gc.cpp */
