/*
   FALCON - The Falcon Programming Language.
   FILE: metaclass.cpp

   Handler for class instances.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 18 Jun 2011 21:24:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classes/classmantra.h>
#include <falcon/error.h>
#include <falcon/mantra.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>

#include <falcon/modspace.h>
#include <falcon/pseudofunc.h>

namespace Falcon {

ClassMantra::ClassMantra():
   Class( "Mantra", FLC_ITEM_USER )
{
}

ClassMantra::ClassMantra( const String& name, int64 type ):
   Class( name, type )
{
}


ClassMantra::~ClassMantra()
{
}

void ClassMantra::gcMarkInstance( void* self, uint32 mark ) const
{
   static_cast<Mantra*>(self)->gcMark( mark );
}

bool ClassMantra::gcCheckInstance( void* self, uint32 mark ) const
{
   return static_cast<Mantra*>(self)->gcCheck( mark );
}

void ClassMantra::dispose( void* self ) const
{
   delete static_cast<Mantra*>(self);
}


void* ClassMantra::clone( void* source ) const
{
   return source;
}

void* ClassMantra::createInstance() const
{
   return 0;
}


void ClassMantra::describe( void* instance, String& target, int, int ) const
{
   Mantra* fc = static_cast<Mantra*>(instance);
   target = "Mantra " + fc->locate();
}

//====================================================================
// Storage
//

void ClassMantra::store( VMContext*, DataWriter* stream, void* instance ) const
{
   Mantra* mantra = static_cast<Mantra*>(instance);
   TRACE1( "ClassMantra::store -- starting store mantra %s", mantra->name().c_ize());

   stream->write( mantra->name() );
   // if store
   if( mantra->module() != 0 )
   {
      stream->write( true );
      stream->write( mantra->module()->name() );
      stream->write( mantra->module()->uri() );
   }
   else {
      stream->write( false );
   }
}


void ClassMantra::restore( VMContext* ctx, DataReader* stream ) const
{
   MESSAGE1( "ClassMantra::restore -- starting restore");
   
   static Engine* eng = Engine::instance()->instance();
   
   String name;
   stream->read(name);      
   TRACE1( "ClassMantra::restore -- Restoring mantra %s", name.c_ize() );
   
   bool bHasModule;      
   stream->read(bHasModule);      
   if( bHasModule ) 
   {
      String modName, modUri;
      stream->read( modName );
      stream->read( modUri );

      TRACE2( "ClassMantra::restore -- Restoring dynamic mantra %s from %s: %s",
            name.c_ize(), modName.c_ize(), modUri.c_ize() );

      //TODO: is the main VM module space the right place?
      ModSpace* ms = ctx->vm()->modSpace();
      // this might alter the context and go deep
      ms->findDynamicMantra( ctx, modUri, modName, name );
   }
   else {
      Mantra* mantra = eng->getMantra( name );
      
      // if 0, would have thrown
      fassert( mantra != 0 );
      ctx->pushData( FALCON_GC_STORE( this, mantra ) );
   }      
}

//====================================================================
// Operator overloads
//


void ClassMantra::op_isTrue( VMContext* ctx, void* ) const
{
   // classes are always true
   ctx->topData().setBoolean(true);
}

void ClassMantra::op_toString( VMContext* ctx , void* item ) const
{
   Class* fc = static_cast<Class*>(item);
   String* sret = new String( "Mantra " );
   sret->append(fc->name());
   ctx->topData() = FALCON_GC_STORE( sret->handler(), sret );
}


}

/* end of metaclass.cpp */
