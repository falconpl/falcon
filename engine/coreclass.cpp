/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.cpp
   $Id: coreclass.cpp,v 1.9 2007/08/11 00:11:53 jonnymind Exp $

   Core Class implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu Jan 20 2005
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
   CoreClass implementation
*/

#include <falcon/cclass.h>
#include <falcon/cobject.h>
#include <falcon/vm.h>

namespace Falcon {

CoreClass::CoreClass( VMachine *origin, Symbol *sym, int modId, PropertyTable *pt ):
   m_modId( modId ),
   m_sym( sym ),
   m_properties( pt ),
   Garbageable( origin, sizeof( this ) + sizeof( void * ) * 2 * pt->size() + sizeof( *pt ) )
{
}

bool CoreClass::derivedFrom( const String &className ) const
{
   // else search the base class name in the inheritance properties.
   uint32 pos;
   if ( m_properties->findKey( &className, pos ) )
   {
      Item *itm =  m_properties->getValue( pos )->dereference();
      if ( itm->isClass() )
         return true;
   }

   return false;
}

CoreClass::~CoreClass()
{
   delete m_properties;
}

CoreObject *CoreClass::createInstance() const
{
   // we must have an origin pointer.
   return new CoreObject( origin(), *m_properties, m_attributes, symbol() );
}

CoreObject *CoreClass::createUncollectedInstance() const
{
   // we must have an origin pointer.
   return new CoreObject( 0, *m_properties, m_attributes, symbol() );
}

}


/* end of coreclass.cpp */
