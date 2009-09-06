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
#include <falcon/memory.h>
#include <falcon/fassert.h>
#include <falcon/mempool.h>

#include <string.h>

namespace Falcon {


//=================================================================================
// The live module class.
//=================================================================================

LiveModule::LiveModule( Module *mod, bool bPrivate ):
   Garbageable(),
   m_module( mod ),
   m_aacc( 0 ),
   m_iacc( 0 ),
   m_bPrivate( bPrivate ),
   m_bAlive(true),
   m_needsCompleteLink( true ),
   m_initState( init_none )
{
   m_module->incref();
   m_strCount = mod->stringTable().size();
   if( m_strCount > 0 )
   {
      m_strings = static_cast<CoreString**>( memAlloc( sizeof(CoreString*) * m_strCount ) );
      memset( m_strings, 0, sizeof(CoreString*) * m_strCount );
   }
   else {
      m_strings = 0;
  }
}


LiveModule::~LiveModule()
{
   fassert( m_module != 0 );
   
   if ( m_strings != 0 )
      memFree( m_strings );

   m_module->decref();
   memPool->accountItems( m_iacc );
   gcMemAccount( m_aacc );
}

void LiveModule::detachModule()
{
   // disengage all the items.
   uint32 i;

   // TODO: memset to 0
   for ( i = 0; i < m_globals.length(); ++i )
   {
      // disengage but not dereferece; we want to nil the globals here,
      // not to destroy the imported symbols.
      m_globals[ i ].setNil();
   }

   for ( i = 0; i < m_wkitems.length(); ++i )
   {
      wkitems()[i].dereference()->setNil();
   }

   m_bAlive = false;
}

Item *LiveModule::findModuleItem( const String &symName ) const
{
   if ( ! isAlive() )
      return 0;

   const Symbol *sym = m_module->findGlobalSymbol( symName );

   if ( sym == 0 )
      return 0;

   return const_cast<Item*>(&m_globals[ sym->itemId() ]);
}

bool LiveModule::finalize()
{
   // resist early destruction
   return false;
}

void LiveModule::gcMark( uint32 mk )
{
   if( mk != mark() )
   {
      mark( mk );

      for( uint32 i = 0; i < m_strCount; ++i )
      {
         if( m_strings[i] != 0 )
            m_strings[i]->mark( mk );
      }

      globals().gcMark( mk );
      wkitems().gcMark( mk );
      userItems().gcMark( mk );
   }
}

String* LiveModule::getString( uint32 stringId ) const
{
   fassert( stringId < (uint32) m_module->stringTable().size() );
   
   if( stringId >= m_strCount )
   {
      m_strings = static_cast<CoreString**>(memRealloc( m_strings, sizeof( CoreString* ) * (stringId+1) ));
      memset( m_strings + m_strCount, 0, sizeof( CoreString* ) * ( stringId - m_strCount +1) );
      m_strCount = stringId+1;
   }
   
   if( m_strings[stringId] == 0 )
   {
      CoreString* dest = new CoreString( *m_module->stringTable().get( stringId ) );
      m_strings[stringId] = dest;
      dest->bufferize();
      gcMemUnaccount( sizeof( CoreString ) + dest->allocated() );
      memPool->accountItems( -1 );
      m_aacc += sizeof( CoreString ) + dest->allocated();
      m_iacc++;
      return dest;
   }

   return m_strings[stringId];
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
