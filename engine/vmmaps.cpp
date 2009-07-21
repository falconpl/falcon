/*
   FALCON - The Falcon Programming Language
   FILE: vmmaps.cpp

   Map items used in VM and related stuff.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 4 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Map items used in VM and related stuff.
*/

#include <falcon/vmmaps.h>
#include <falcon/livemodule.h>
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
   SymModuleTraits &t_symmodule() { static SymModuleTraits dt; return dt; }
}

SymModuleMap::SymModuleMap():
   Map( &traits::t_stringptr(), &traits::t_symmodule() )
{}


SymModule::SymModule( LiveModule *mod, const Symbol *sym ):
   m_item( &mod->globals()[ sym->itemId() ] ),
   m_symbol( sym ),
   m_lmod( mod ),
   m_wkiid( -1 )
{}


//==================================================================
// Live module map
//

namespace traits
{
   LiveModulePtrTraits &t_livemoduleptr() { static LiveModulePtrTraits dt; return dt; }
}

LiveModuleMap::LiveModuleMap():
   Map( &traits::t_string(), &traits::t_livemoduleptr() )
{}

}

/* end of vmmaps.cpp */
