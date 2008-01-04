/*
   FALCON - The Falcon Programming Language
   FILE: vmmaps.cpp

   Map items used in VM and related stuff.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 4 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Map items used in VM and related stuff.
*/

#include <falcon/vmmaps.h>
#include <string.h>

namespace Falcon {

uint32 SymModuleTraits::memSize() const
{
   return sizeof( SymModule );
}

void SymModuleTraits::init( void *itemZone ) const
{
   // nothing
}

void SymModuleTraits::copy( void *targetZone, const void *sourceZone ) const
{
   SymModule *target = (SymModule *) targetZone;
   SymModule *source = (SymModule *) sourceZone;

   memcpy( target, source, sizeof( SymModule ) );
}

int SymModuleTraits::compare( const void *firstz, const void *secondz ) const
{
   // Never used as key
   return 0;
}

void SymModuleTraits::destroy( void *item ) const
{
   // do nothing
}

bool SymModuleTraits::owning() const
{
   return false;
}

namespace traits
{
   SymModuleTraits t_symmodule;
}

SymModuleMap::SymModuleMap():
   Map( &traits::t_stringptr, &traits::t_symmodule )
{}



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

namespace traits
{
   LiveModulePtrTraits t_livemoduleptr;
}

LiveModuleMap::LiveModuleMap():
   Map( &traits::t_stringptr, &traits::t_livemoduleptr )
{}


LiveModule::LiveModule( VMachine *vm, Module *mod ):
   Garbageable( vm, sizeof( *this ) ),
   m_module( mod )
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
      m_module->decref();
      m_module = 0;
      // no reason to keep globals allocated
      m_globals.resize(0);
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

}

/* end of vmmaps.cpp */
