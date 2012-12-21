/*
   FALCON - The Falcon Programming Language.
   FILE: famloader.cpp

   Precompiled module deserializer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/famloader.cpp"

#include <falcon/famloader.h>
#include <falcon/modspace.h>
#include <falcon/stream.h>
#include <falcon/errors/ioerror.h>
#include <falcon/restorer.h>
#include <falcon/module.h>
#include <falcon/process.h>
#include <falcon/errors/ioerror.h>
#include <falcon/trace.h>

#include <falcon/vmcontext.h>

namespace Falcon
{

FAMLoader::FAMLoader( ModSpace* ms ):
   m_modSpace( ms ),
   m_stepLoad(this)
{}

FAMLoader::~FAMLoader()
{}
   
void FAMLoader::load( VMContext* ctx, Stream* ins , const String& path, const String& name )
{
   static Class* streamClass = Engine::instance()->streamClass();
   static Class* restClass = Engine::instance()->restorerClass();

   char buffer[4];
   ins->read( buffer, sizeof(buffer) );

   if( buffer[0] != 'F' || buffer[1] != 'M' )
   {
      throw new IOError( ErrorParam( e_mod_not_fam, __LINE__, SRC ) 
         .origin( ErrorParam::e_orig_loader )
         .extra(path) );
   }
   else if ( buffer[2] < 4 || buffer[3] < 1 )
   {
      throw new IOError( ErrorParam( e_mod_unsupported_fam, __LINE__, SRC ) 
         .origin( ErrorParam::e_orig_loader )
         .extra(path) );
   }

   Restorer* restorer = new Restorer;

   ctx->pushData( Item(restClass, restorer) );
   ctx->pushData( Item(streamClass, ins) );
   ctx->pushData( FALCON_GC_HANDLE(new String(path)) );
   ctx->pushData( FALCON_GC_HANDLE(new String(name)));

   ctx->pushCode( &m_stepLoad );
}


void FAMLoader::PStepLoad::apply_( const PStep* self, VMContext* ctx )
{
   static Class* modClass = Engine::instance()->moduleClass();
   const FAMLoader::PStepLoad* pstep = static_cast<const FAMLoader::PStepLoad*>( self );
   int32 &seqId = ctx->currentCode().m_seqId;

   TRACE("FAMLoader::PStepLoad::apply_ %d", seqId );

   Restorer* restorer = static_cast<Restorer*>(ctx->opcodeParam(3).asInst());
   Stream* ins = static_cast<Stream*>(ctx->opcodeParam(2).asInst());

   switch( seqId )
   {
   case 0:
      seqId++;
      restorer->restore( ctx, ins, pstep->m_owner->m_modSpace );
      MESSAGE1("FAMLoader::PStepLoad::apply_ trying again..." );
      return;
   }

   Class* handler = 0;
   void* data = 0;
   bool first = 0;

   String& path = *static_cast<String*>(ctx->opcodeParam(1).asInst());
   String& name = *static_cast<String*>(ctx->opcodeParam(0).asInst());

   if( ! restorer->next( handler, data, first ) || handler != modClass )
   {
      throw new IOError( ErrorParam( e_mod_not_fam, __LINE__, SRC )
              .origin( ErrorParam::e_orig_loader )
              .extra(path) );
   }

   Module* mod = static_cast<Module*>( data );
   mod->name( name );
   mod->uri( path );

   TRACE("FAMLoader::PStepLoad::apply_ %s:%s restore complete", name.c_ize(), path.c_ize() );

   ctx->stackResult(4, FALCON_GC_STORE(modClass, mod));
   ctx->popCode();
}

}

/* end of famloader.cpp */
