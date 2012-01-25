/*
   FALCON - The Falcon Programming Language.
   FILE: classfunction.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classsfunction.cpp"

#include <falcon/trace.h>
#include <falcon/classes/classfunction.h>
#include <falcon/synfunc.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/module.h>
#include <falcon/pseudofunc.h>

#include <falcon/vm.h>
#include <falcon/modspace.h>
#include <falcon/modloader.h>

#include <falcon/errors/ioerror.h>

namespace Falcon {

ClassFunction::ClassFunction():
   Class("Function", FLC_CLASS_ID_FUNC )
{
}


ClassFunction::~ClassFunction()
{
}

void ClassFunction::dispose( void* self ) const
{
   Function* f = static_cast<Function*>(self);
   delete f;
}


void* ClassFunction::clone( void* ) const
{
   //Function* f = static_cast<Function*>(self);
   //TODO
   return 0;
}


void ClassFunction::store( VMContext* ctx, DataWriter* stream, void* instance ) const
{
   Function* func = static_cast<Function*>(instance);
   TRACE1( " -- starting store function %s", func->name().c_ize());

   MetaStorer* ms = func->metaStorer();
   
   if( ms == 0 )
   {
      // just store the name
      stream->write(false);
      // if store
      if( func->module() != 0 )
      {
         stream->write( true );
         stream->write( func->module()->name() );
         stream->write( func->module()->uri() );
      }
      else {
         stream->write( false );
      }
      
      stream->write( func->name() );
   }
   else {
      stream->write( true );
      stream->write( ms->name() );
      ms->store(ctx, stream, instance );
   }   
}


void ClassFunction::restore( VMContext* ctx, DataReader* stream, void*& empty ) const
{
   MESSAGE1( " -- starting restore");
   
   static Engine* eng = Engine::instance()->instance();
   
   bool bHasStorer;
   stream->read( bHasStorer );
   if( bHasStorer )
   {
      String name;
      stream->read(name);
      
      TRACE2( " -- Restoring function via MetaStorer %s", name.c_ize() );
      MetaStorer* ms = eng->getMetaStorer(name);
      if( ms == 0 ) {
         //throw
         fassert(false);
      }
      
      ms->restore( ctx, stream, empty );
   }
   else 
   {
      bool bHasModule;
      String funcName;
      
      stream->read(bHasModule);      
      if( bHasModule ) 
      {
         bool hasLink = false;
         
         String modName, modUri;
         stream->read( modName );
         stream->read( modUri );
         stream->read( funcName );
         
         TRACE2( " -- Restoring dynamic function %s from %s: %s", 
               funcName.c_ize(), modName.c_ize(), modUri.c_ize() );
         
         ModSpace* ms = ctx->vm()->modSpace();
         ModLoader* ml = ctx->vm()->modLoader();
         Function* func = ms->findDynamicFunction(ml, modUri, modName, funcName, hasLink );
         // if 0, would have thrown
         fassert( func != 0 );
         
         // honor link request.
         if( hasLink )
         {
            Error* err = ms->link();
            if( err != 0 ) throw err;            
            ms->readyVM( ctx );
         }
         
         empty = func;
      }
      else {
         Function* func = eng->getPseudoFunction( funcName );
         empty = func;
      }      
   }

}


void ClassFunction::flatten( VMContext* ctx, ItemArray& items, void* instance ) const
{
   Function* func = static_cast<Function*>(instance);
   TRACE1( " -- starting flattening function %s", func->name().c_ize() );
   
   MetaStorer* ms = func->metaStorer();   
   if( ms != 0 )
   {
      TRACE2( " -- Function %s has a meta storer -- flattening", 
               func->name().c_ize() );
      ms->flatten(ctx, items, instance);
   }
}


void ClassFunction::unflatten( VMContext* ctx, ItemArray& items, void* instance ) const
{
   Function* func = static_cast<Function*>(instance);
   TRACE1( " -- starting unflattening function %s", func->name().c_ize() );
   
   MetaStorer* ms = func->metaStorer();   
   if( ms != 0 )
   {
      TRACE2( " -- Function %s has a meta storer -- unflattening", 
               func->name().c_ize() );
      ms->unflatten(ctx, items, instance);
   }
}


void ClassFunction::describe( void* instance, String& target, int, int ) const
{
   Function* func = static_cast<Function*>(instance);
   target = func->name() + "()";
}


void ClassFunction::gcMark( void* self, uint32 mark ) const
{
   static_cast<Function*>(self)->gcMark(mark);
}


bool ClassFunction::gcCheck( void* self, uint32 mark ) const
{
   return static_cast<Function*>(self)->gcCheck(mark);
}



void ClassFunction::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   ctx->call( static_cast<Function*>(self), paramCount );
}


void ClassFunction::op_eval( VMContext* ctx, void* self ) const
{
   // called object is on top of the stack 
   ctx->call( static_cast<Function*>(self), 0 );
}

}

/* end of classfunction.cpp */
