/*
   FALCON - The Falcon Programming Language.
   FILE: classrestorer.cpp

   Falcon core module -- Interface to Restorer class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 07 Dec 2011 23:31:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/classes/classrestorer.cpp"

#include <falcon/classes/classrestorer.h>
#include <falcon/restorer.h>
#include <falcon/datareader.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>

#include <falcon/module.h>
#include <falcon/modspace.h>

#include <falcon/classes/classstream.h>

namespace Falcon {


//====================================================
// Properties.
//

static void  get_hasNext( const Class*, const String &, void* instance, Item& value )
{ 
   Restorer* restorer = static_cast<Restorer*>(instance);
   value.setBoolean( restorer->hasNext() );
}


FALCON_DECLARE_FUNCTION( next, "" );
void Function_next::invoke( VMContext* ctx, int32 )
{  
   Restorer* restorer = static_cast<Restorer*>(ctx->self().asInst());
   Class* cls;
   void* data;
   bool first;
   if( restorer->next( cls, data, first ) )
   {
      if (cls->isFlatInstance())
      {
         ctx->returnFrame( *static_cast<Item*>(data) );
      }
      else {
         // Never send to the garbage, the restorer knows how to do that.
         ctx->returnFrame( Item(cls, data) );
      }
   }
   else
   {
      ctx->raiseError(new CodeError( ErrorParam(e_arracc, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime )
         .extra( "No more items in next()") ) );
   }
}

FALCON_DECLARE_FUNCTION( restore, "stream:Stream" );
void Function_restore::invoke( VMContext* ctx, int32 )
{
   static Class* clsStream = Engine::handlers()->streamClass();
   static PStep* retStep = &Engine::instance()->stdSteps()->m_returnFrame;
   
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
   
   Restorer* restorer = static_cast<Restorer*>(ctx->self().asInst());
   Stream* streamc = static_cast<Stream*>(cls->getParentData(clsStream,data));

   // prepare not to return the frame now but later.
   ctx->pushCode( retStep );
   restorer->restore(ctx, streamc, ctx->process()->modSpace() );
}


ClassRestorer::ClassRestorer():
   Class("Restorer")
{
   addProperty( "hasNext", &get_hasNext );
   addMethod( new Function_next );
   addMethod( new Function_restore );
}


ClassRestorer::~ClassRestorer()
{}

bool ClassRestorer::op_init( VMContext* , void*, int32 ) const
{
   return false;
}


void* ClassRestorer::createInstance() const
{ 
   return new Restorer;
}

void ClassRestorer::dispose( void* instance ) const
{
   delete static_cast<Restorer*>(instance);
}

void* ClassRestorer::clone( void* instance ) const
{
   return new Restorer( *static_cast<Restorer*>(instance) );
}

void ClassRestorer::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<Restorer*>(instance)->reader().gcMark(mark);
}

bool ClassRestorer::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<Restorer*>(instance)->reader().gcMark() >= mark ;
}


/*
void ClassRestorer::op_iter( VMContext* ctx, void* instance ) const
{
   
}


void ClassRestorer::op_next( VMContext* ctx, void* instance ) const
{
   
}
*/

}

/* end of classrestorer.cpp */
