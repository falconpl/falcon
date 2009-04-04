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

#include <string.h>

namespace Falcon {


//=================================================================================
// The live module class.
//=================================================================================

LiveModule::LiveModule( Module *mod, bool bPrivate ):
   Garbageable(),
   m_module( mod ),
   m_bPrivate( bPrivate ),
   m_strings(0),
   m_initState( init_none )
{
   m_module->incref();
   m_stringCount = m_module->stringTable().size();

   if ( m_stringCount > 0 )
   {
      m_strings = (String**) memAlloc( sizeof(String *) * m_stringCount );
      memset( m_strings, 0, m_stringCount * sizeof(String *) );
   }

}


LiveModule::~LiveModule()
{
   fassert( m_module != 0 );
   m_module->decref();

   if ( m_strings != 0 )
   {
      for( uint32 i = 0; i < m_stringCount; ++i )
      {
         if ( m_strings[i] != 0 )
            delete m_strings[i];
      }

      memFree( m_strings );
   }
}

void LiveModule::detachModule()
{
   // disengage all the items.
   uint32 i;

   for ( i = 0; i < m_globals.size(); ++i )
   {
      m_globals.itemAt( i ).dereference()->setNil();
   }

   for ( i = 0; i < m_wkitems.size(); ++i )
   {
      wkitems().itemAt(i).dereference()->setNil();
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

String* LiveModule::getString( uint32 stringId ) const
{
   //return (String*)m_module->stringTable().get( stringId );

   if ( stringId >= m_stringCount )
   {
      String *dest;
      uint32 size;

      fassert( m_module != 0 );
      dest = new String( *m_module->stringTable().get( stringId ) );
      dest->liveModule( this );
      size = m_module->stringTable().size();
      // this may (legally) happen only when m_module is a flexy module
      // and its table has been grown in the meanwhile
      fassert( size > stringId );
      const_cast<LiveModule*>(this)->m_strings = (String**) memRealloc( m_strings, sizeof(String *) * size );
      memset( m_strings + m_stringCount, 0, (size - m_stringCount) * sizeof(String *) );

      const_cast<LiveModule*>(this)->m_stringCount = size;
      m_strings[stringId] = dest;
   }
   else if( m_strings[stringId] == 0 )
   {
      String* dest = new String( *m_module->stringTable().get( stringId ) );
      dest->liveModule( this );
      m_strings[stringId] = dest;
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
