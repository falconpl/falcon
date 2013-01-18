/*
   FALCON - The Falcon Programming Language.
   FILE: create.cpp

   Interface to Falcon Virtual File System -- create() function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 04 Dec 2011 11:09:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/vfs/create.cpp"

#include <falcon/vmcontext.h>
#include <falcon/vfsprovider.h>
#include <falcon/engine.h>
#include <falcon/cm/uri.h>
#include <falcon/classes/classstream.h>
#include <falcon/stream.h>

#include "vfs.h"
#include "create.h"

namespace Falcon {
namespace Ext {

Create::Create( VFSModule* mod ):
   Function("create"),
   m_module( mod )
{ 
   parseDescription("uri:S|URI,mode:[N]"); 
}
      
Create::~Create() {}
   
void Create::invoke( Falcon::VMContext* ctx, int )
{
   static Engine* inst = Engine::instance();
   
   Item* i_uri = ctx->param(0);
   Item* i_mode = ctx->param(1);
   
   Class* cls; void* data; 
   bool isString = false;
   if ( ! i_uri->asClassInst( cls, data ) || 
        ! ((isString = i_uri->isString()) || cls->isDerivedFrom(m_module->uriClass()))
        || (i_mode != 0 && i_mode->isInteger())
      )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }
   
   VFSProvider::CParams op;
   
   if( i_mode != 0 )
   {
      op = VFSProvider::CParams(i_mode->forceInteger());
   }
   else
   {
      op.wrOnly();
   }
   
   Stream* stream;
   if( isString )
   {
      URI theUri( *static_cast<String*>(data) );
      stream = inst->vfs().create( theUri, op );
   }
   else
   {
      URICarrier* uricar = static_cast<URICarrier*>(cls->getParentData( m_module->uriClass(), data ));
      URI& theUri = uricar->m_uri;
      stream = inst->vfs().create( theUri, op );
   }
   
   stream->shouldThrow(true);
   StreamCarrier* scr = new StreamCarrier(stream);
   if( i_mode == 0 || ((i_mode->forceInteger() & FALCON_VFS_MODE_FLAG_RAW) == 0) )
   {
      scr->setBuffering(4096);
   }

   ctx->returnFrame( FALCON_GC_STORE( m_module->streamClass(), scr ) );
}

}
}

/* end of create.cpp */
