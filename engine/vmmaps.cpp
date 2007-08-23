/*
   FALCON - The Falcon Programming Language
   FILE: vmmaps.cpp
   $Id: vmmaps.cpp,v 1.2 2006/11/06 21:06:04 gian Exp $

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
   SymModule *first = (SymModule *) firstz;
   SymModule *second = (SymModule *) secondz;

   if ( first->moduleId() < second->moduleId() )
      return -1;
   if ( first->moduleId() > second->moduleId() )
      return 1;

   if ( first->symbolId() < second->symbolId() )
      return -1;
   if ( first->symbolId() > second->symbolId() )
      return 1;

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

}

/* end of vmmaps.cpp */
