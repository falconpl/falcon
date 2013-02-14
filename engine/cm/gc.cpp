/*
   FALCON - The Falcon Programming Language.
   FILE: gc.cpp

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/gc.cpp"

#include <falcon/cm/gc.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {
namespace Ext {


ClassGC::ClassGC():
   ClassUser("%GC"),
   FALCON_INIT_PROPERTY( memory ),
   FALCON_INIT_PROPERTY( items ),
   FALCON_INIT_PROPERTY( enabled )
{
   // we don't need an object
   m_bIsFlatInstance = true;
}

ClassGC::~ClassGC()
{}


void* ClassGC::createInstance() const
{
   return 0;
}

void ClassGC::dispose( void* ) const
{
   // nothing to do
}

void* ClassGC::clone( void* instance ) const
{
   return instance;
}


void ClassGC::gcMarkInstance( void*, uint32 ) const
{
   // nothing to do
}

bool ClassGC::gcCheckInstance( void*, uint32 ) const
{
   // nothing to do
   return true;
}

bool ClassGC::op_init( VMContext* , void*, int  ) const
{
   // nothing to do
   return true;
}

//====================================================
// Properties.
//
   
FALCON_DEFINE_PROPERTY_SET_P0( ClassGC, memory )
{
   throw new CodeError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("memory") );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, memory )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->storedMemory();
}

FALCON_DEFINE_PROPERTY_SET_P0( ClassGC, items )
{
   throw new ParamError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra("items") );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, items )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->storedItems();
}

FALCON_DEFINE_PROPERTY_SET( ClassGC, enabled )(void*, const Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   coll->enable( value.isTrue() );
}

FALCON_DEFINE_PROPERTY_GET( ClassGC, enabled )(void*, Item& value)
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->isEnabled();
}

}
}

/* end of gc.cpp */
