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
#include <falcon/module.h>
#include <falcon/stdsteps.h>

#include <falcon/cm/stream.h>

namespace Falcon {
namespace Ext {

class StorerCarrier: public UserCarrierT<Storer> 
{
   
public:
   StorerCarrier( Storer* data ):
      UserCarrierT<Storer> (data),
      m_streamc(0)
   {}
      
   virtual ~StorerCarrier()
   {
   }
   
   void setStream( StreamCarrier* stc )
   {     
      m_streamc = stc;
   }
   
   virtual void gcMark( uint32 mark )
   {
      if ( m_gcMark != mark )
      {
         m_gcMark = mark;
         m_streamc->m_gcMark = mark;
      }
   }
   
private:
   StreamCarrier* m_streamc;
};


ClassStorer::ClassStorer():
   ClassUser("Storer"),
   FALCON_INIT_METHOD( store ),
   FALCON_INIT_METHOD( addFlatMantra ),
   FALCON_INIT_METHOD( commit )
{}

ClassStorer::~ClassStorer()
{}

bool ClassStorer::op_init( VMContext* ctx,  void* instance, int32 ) const
{
   StorerCarrier* sc = static_cast<StorerCarrier*>(instance);
   sc->carried()->context( ctx );
   return false;
}


void* ClassStorer::createInstance() const
{ 
   return new StorerCarrier( new Storer );
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
   
   Storer* storer = static_cast<StorerCarrier*>(ctx->self().asInst())->carried();
   Class* cls; void *data; 
   i_item->forceClassInst( cls, data );
   
   // prepare an explicit call of the return frame
   ctx->pushCode( &stdSteps->m_returnFrame );
   
   // we must return only if the store was completed in this loop
   if( storer->store( cls, data ) )
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
   
   Storer* storer = static_cast<StorerCarrier*>(ctx->self().asInst())->carried();
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
   static Class* clsStream = methodOf()->module()->getClass( "Stream" );
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
   
   StorerCarrier* stc = static_cast<StorerCarrier*>(ctx->self().asInst());
   Storer* storer = stc->carried();
   StreamCarrier* streamc = static_cast<StreamCarrier*>(data);
   stc->setStream( streamc );
   
   // prepare an explicit call of the return frame
   ctx->pushCode( &stdSteps->m_returnFrame );
   
   // skip internal buffering, even if provided, by taking the underlying
   if( storer->commit( streamc->m_underlying ) )
   {
      // we must return only if the store was completed in this loop
      ctx->returnFrame();
   }
}

}
}

/* end of storer.cpp */
