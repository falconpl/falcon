/*
   FALCON - The Falcon Programming Language.
   FILE: storer.cpp

   Falcon core module -- Interface to Storer class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/storer.cpp"

#include <falcon/cm/storer.h>
#include <falcon/storer.h>
#include <falcon/datawriter.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/usercarrier.h>

#include "falcon/module.h"

namespace Falcon {
namespace Ext {

typedef UserCarrierT<Storer> StorerCarrier;

ClassStorer::ClassStorer():
   ClassUser("Storer"),
   FALCON_INIT_METHOD( store ),
   FALCON_INIT_METHOD( commit )
{}

ClassStorer::~ClassStorer()
{}

void ClassStorer::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector(); 
   
   void* instance = new StorerCarrier( new Storer(ctx) );
   ctx->stackResult( pcount + 1, FALCON_GC_STORE( coll, this, instance ) );
}


void* ClassStorer::createInstance( Item*, int  ) const
{ 
   return 0;
}

//====================================================
// Properties.
//
   

FALCON_DEFINE_METHOD_P1( ClassStorer, store )
{
   Item* i_item = ctx->param(0);
   if( i_item == 0 )
   {
      throw paramError();
   }
   
   Storer* storer = static_cast<StorerCarrier*>(ctx->self().asInst())->carried();
   Class* cls; void *data; 
   i_item->forceClassInst( cls, data );
   storer->store( cls, data );
   ctx->returnFrame();   
}


FALCON_DEFINE_METHOD_P1( ClassStorer, commit )
{  
   static Class* clsStream = module()->getClass( "Stream" );
   fassert( clsStream != 0 );
   
   Item* i_item = ctx->param(0);
   if( i_item == 0 )
   {
      throw paramError();
   }

   Class* cls; void* data;
   i_item->forceClassInst( cls, data ); 
   
   if( ! cls->isDerivedFrom( clsStream ) )
   {
      throw paramError();
   }
   
   Storer* storer = static_cast<StorerCarrier*>(ctx->self().asInst())->carried();
   DataWriter dw( static_cast<Stream*>(data) );
   storer->commit( &dw );
   ctx->returnFrame();   
}

}
}

/* end of storer.cpp */
