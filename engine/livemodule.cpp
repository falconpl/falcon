/*
   FALCON - The Falcon Programming Language
   FILE: livemodule.cpp

   The Representation of module live data once linked in a VM
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 03 Apr 2009 23:27:53 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   MThe Representation of module live data once linked in a VM
*/

#include <falcon/livemodule.h>
#include <string.h>

namespace Falcon {


//=================================================================================
// The live module class.
//=================================================================================

LiveModule::LiveModule( Module *mod, bool bPrivate ):
   Garbageable(),
   m_module( mod ),
   m_bPrivate( bPrivate ),
   m_initState( init_none )
{
   m_module->incref();
}


LiveModule::~LiveModule()
{
   if ( m_module != 0 )
      m_module->decref();
}

void LiveModule::detachModule()
{
   if ( m_module != 0 )
   {
      Module *m = m_module;
      m_module = 0;
      // no reason to keep globals allocated
      m_globals.resize(0);
      m_wkitems.resize(0);

      m->decref();
   }
}

Item *LiveModule::findModuleItem( const String &symName ) const
{
   if ( ! isAlive() )
      return 0;

   const Symbol *sym = m_module->findGlobalSymbol( symName );

   if ( sym == 0 )
      return 0;

   return m_globals.itemPtrAt( sym->itemId() );
}

bool LiveModule::finalize()
{
   // resist early destruction
   return false;
}

//=================================================================================
// Live module related traits
//=================================================================================

uint32 LiveModulePtrTraits::memSize() const
{
   return sizeof( LiveModule * );
}

void LiveModulePtrTraits::init( void *itemZone ) const
{
   itemZone = 0;
}

void LiveModulePtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   LiveModule **target = (LiveModule **) targetZone;
   LiveModule *source = (LiveModule *) sourceZone;

   *target = source;
}

int LiveModulePtrTraits::compare( const void *firstz, const void *secondz ) const
{
   // never used as key

   return 0;
}

void LiveModulePtrTraits::destroy( void *item ) const
{
   /* Owned by GC
   LiveModule *ptr = *(LiveModule **) item;
   delete ptr;
   */
}

bool LiveModulePtrTraits::owning() const
{
   /* Owned by GC */
   return false;
}


}

/* end of livemodule.cpp */
