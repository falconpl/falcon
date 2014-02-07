/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: container.cpp
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#include "classcontainer.h"
#include "container_mod.h"
#include "iterator_mod.h"
#include "containers_fm.h"

#include <falcon/vmcontext.h>

namespace Falcon {
namespace Mod {
Container::Container( const ClassContainerBase* h ):
         m_handler(h)
{}

Container::~Container()
{}


bool Container::contains( VMContext* ctx, const Item& value )
{
   ModuleContainers* mc = static_cast<ModuleContainers*>(m_handler->module());
   Iterator* iter = iterator();

   ctx->pushData( value );
   ctx->pushData( FALCON_GC_STORE(mc->iteratorClass(), iter ) );
   ctx->pushData(Item((int64)1));
   ctx->pushCode(m_handler->stepContains());

   return false;
}


void Container::gcMark( uint32 m )
{
   if( m == m_mark )
   {
      return;
   }

   m_mark = m;
   Iterator* iter = iterator();

   lock();
   int32 v = version();
   Item current;
   while( iter->next(current, false) )
   {
      fassert( current != 0 );
      unlock();

      try {
         current.gcMark( m );
      }
      catch(...)
      {
         delete iter;
         throw;
      }

      lock();
      if( v != version() )
      {
         iter->reset();
      }
   }
   unlock();

   delete iter;
}
}
}

/* end of container.cpp */
