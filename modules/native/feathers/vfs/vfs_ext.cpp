/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_ext.h

   Interface to Falcon Virtual File System -- various
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 21 Mar 2013 22:23:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#define SRC "falcon/modules/native/feathers/vfs/vfs_ext.cpp"

#include <falcon/vmcontext.h>
#include <falcon/vfsprovider.h>
#include <falcon/engine.h>
#include <falcon/stream.h>
#include <falcon/classes/classstream.h>

#include "vfs_ext.h"

namespace Falcon {
namespace Ext {

void Function_InputStream::invoke( Falcon::VMContext* ctx, int )
{
   static Engine* inst = Engine::instance();

   Item* i_uri = ctx->param(0);
   
   if ( ! i_uri->isString() )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }
   
   VFSProvider::OParams op;
   op.rdOnly();
   
   Stream* stream;
   URI theUri( *i_uri->asString() );
   stream = inst->vfs().open( theUri, op );
   
   stream->shouldThrow(true);
   ctx->returnFrame( FALCON_GC_HANDLE( stream ) );
}


void Function_OutputStream::invoke( Falcon::VMContext* ctx, int )
{
   static Engine* inst = Engine::instance();

   Item* i_uri = ctx->param(0);

   if ( ! i_uri->isString() )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   VFSProvider::CParams op;
   op.rdOnly();

   Stream* stream;
   URI theUri( *i_uri->asString() );
   stream = inst->vfs().create( theUri, op );

   stream->shouldThrow(true);
   ctx->returnFrame( FALCON_GC_HANDLE( stream ) );
}

void Function_IOStream::invoke( Falcon::VMContext* ctx, int )
{
   static Engine* inst = Engine::instance();

   Item* i_uri = ctx->param(0);

   if ( ! i_uri->isString() )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   Stream* stream;

   try
   {
      VFSProvider::CParams op;
      op.rdwr();
      op.noOvr();
      op.append();

      URI theUri( *i_uri->asString() );
      stream = inst->vfs().create( theUri, op );
   }
   catch(...)
   {
      VFSProvider::CParams op;
      op.rdwr();
      op.noOvr();
      op.append();

      URI theUri( *i_uri->asString() );
      stream = inst->vfs().open( theUri, op );
   }

   stream->shouldThrow(true);
   ctx->returnFrame( FALCON_GC_HANDLE( stream ) );
}

}
}

/* end of vfs_ext.cpp */
