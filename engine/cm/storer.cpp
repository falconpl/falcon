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

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/cm/uri.h>

namespace Falcon {
namespace Ext {


ClassStorer::ClassStorer():
   ClassUser("Storer"),
   FALCON_INIT_METHOD( store ),
   FALCON_INIT_METHOD( commit )
{
   
}

ClassStorer::~ClassStorer()
{}

void* ClassStorer::createInstance( Item* params, int pcount ) const
{ 
   return new Storer;
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
   
   Storer* storer = static_cast<Storer*>(ctx->self().asInst());
   
   ctx->returnFrame();   
}


FALCON_DEFINE_METHOD_P1( ClassPath, relativize )
{
   Item* i_path = ctx->param(0);
   if( i_path == 0 || ! i_path->isString() )
   {
      throw paramError();
   }
   
   PathCarrier* pc = static_cast<PathCarrier*>(ctx->self().asInst());
   Item ret; 
   ret.setBoolean( pc->m_path.relativize( *i_path->asString() ) );
   ctx->returnFrame(ret);   
}


FALCON_DEFINE_METHOD_P1( ClassPath, canonicize )
{
   PathCarrier* pc = static_cast<PathCarrier*>(ctx->self().asInst());
   pc->m_path.canonicize();
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassPath, cwd )
{   
   String temp;
   Path::currentWorkDirectory( temp );
   ctx->returnFrame( temp );
}


}
}

/* end of storer.cpp */
