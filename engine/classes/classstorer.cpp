/*
   FALCON - The Falcon Programming Language.
   FILE: classstorer.cpp

   Falcon core module -- Interface to Storer class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/classes/classstorer.cpp"

#include <falcon/classes/classstorer.h>
#include <falcon/storer.h>
#include <falcon/datawriter.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/usercarrier.h>
#include <falcon/module.h>
#include <falcon/stdsteps.h>
#include <falcon/classes/classstream.h>

namespace Falcon {

ClassStorer::ClassStorer():
   ClassUser("Storer"),
   FALCON_INIT_METHOD( store ),
   FALCON_INIT_METHOD( addFlatMantra ),
   FALCON_INIT_METHOD( commit )
{}

ClassStorer::~ClassStorer()
{}

bool ClassStorer::op_init( VMContext*,  void*, int32 ) const
{
   return false;
}


void* ClassStorer::createInstance() const
{ 
   return new Storer;
}

void ClassStorer::dispose( void* instance ) const
{
   delete static_cast<Storer*>(instance);
}

void* ClassStorer::clone( void* instance ) const
{
   return new Storer( *static_cast<Storer*>(instance) );
}

void ClassStorer::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<Storer*>(instance)->writer().gcMark(mark);
}

bool ClassStorer::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<Storer*>(instance)->writer().gcMark() >= mark ;
}


//====================================================
// Properties.
//
   

FALCON_DEFINE_METHOD_P1( ClassStorer, store )
{
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   
   Item* i_item = ctx->param(0);
   if( i_item == 0 )
   {
      throw paramError();
   }
   
   Storer* storer = static_cast<Storer*>(ctx->self().asInst());
   Class* cls; void *data; 
   i_item->forceClassInst( cls, data );
   
   // prepare an explicit call of the return frame
   ctx->pushCode( &stdSteps->m_returnFrame );
   
   // we must return only if the store was completed in this loop
   if( storer->store( ctx, cls, data ) )
   {
      ctx->returnFrame();
   }
}


FALCON_DEFINE_METHOD_P1( ClassStorer, addFlatMantra )
{
   static Class* clsMantra = Engine::instance()->mantraClass();
   
   Item* i_item = ctx->param(0);
   if( i_item == 0 )
   {
      throw paramError();
   }
   
   Storer* storer = static_cast<Storer*>(ctx->self().asInst());
   Class* cls; void *data; 
   i_item->forceClassInst( cls, data );
   
   if( ! cls->isDerivedFrom( clsMantra ) )
   {
      throw paramError();
   }
   
   storer->addFlatMantra( static_cast<Mantra*>(cls->getParentData( clsMantra, data )) );
}


FALCON_DEFINE_METHOD_P1( ClassStorer, commit )
{  
   static StdSteps* stdSteps = Engine::instance()->stdSteps();
   static Class* clsStream = Engine::instance()->streamClass();
   
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
   
   Storer* storer = static_cast<Storer*>(ctx->self().asInst());
   Stream* streamc = static_cast<Stream*>(data);

   // prepare an explicit call of the return frame
   ctx->pushCode( &stdSteps->m_returnFrame );
   
   // skip internal buffering, even if provided, by taking the underlying
   if( storer->commit( ctx, streamc ) )
   {
      // we must return only if the store was completed in this loop
      ctx->returnFrame();
   }
}

}

/* end of classstorer.cpp */
