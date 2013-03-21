/*
   FALCON - The Falcon Programming Language.
   FILE: open.cpp

   Interface to Falcon Virtual File System -- open() function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#define SRC "falcon/modules/native/feathers/vfs/open.cpp"

#include <falcon/vmcontext.h>
#include <falcon/vfsprovider.h>
#include <falcon/engine.h>
#include <falcon/cm/uri.h>
#include <falcon/stream.h>
#include <falcon/classes/classstream.h>

#include "vfs.h"
#include "open.h"

namespace Falcon {
namespace Ext {

Open::Open( VFSModule* mod ):
   Function("open"),
   m_module( mod )
{ 
   parseDescription("uri:S|URI,mode:[N],shmode:[N]"); 
}
      
Open::~Open() {}
   
void Open::invoke( Falcon::VMContext* ctx, int )
{
   static Engine* inst = Engine::instance();
   
   Item* i_uri = ctx->param(0);
   Item* i_mode = ctx->param(1);
   Item* i_shmode = ctx->param(2);
   
   Class* cls; void* data; 
   bool isString = false;
   if ( ! i_uri->asClassInst( cls, data ) || 
        ! ((isString = i_uri->isString()) || cls->isDerivedFrom(m_module->uriClass()))
        || (i_mode != 0 && (i_mode->isInteger()||i_mode->isNil()))
        || (i_shmode != 0 && i_shmode->isInteger())
      )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }
   
   VFSProvider::OParams op;
   
   if( i_mode != 0 && ! i_mode->isNil() )
   {
      op = VFSProvider::OParams((int32)i_mode->forceInteger());
   }
   else
   {
      op.rdOnly();
   }
   
   if( i_shmode != 0 )
   {
      int64 shMode = i_shmode->asInteger();
      if( shMode & VFSProvider::OParams::e_sflag_nr )
      {
         op.shNoRead();
      }
      if( shMode & VFSProvider::OParams::e_sflag_nw )
      {
         op.shNoWrite();
      }
   }
   
   
   Stream* stream;
   if( isString )
   {
      URI theUri( *static_cast<String*>(data) );
      stream = inst->vfs().open( theUri, op );
   }
   else
   {
      URICarrier* uricar = static_cast<URICarrier*>(cls->getParentData( m_module->uriClass(), data ));
      URI& theUri = uricar->m_uri;
      stream = inst->vfs().open( theUri, op );
   }
   
   stream->shouldThrow(true);
   ctx->returnFrame( FALCON_GC_HANDLE( stream ) );
}

}
}

/* end of open.cpp */
