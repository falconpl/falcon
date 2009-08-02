/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.cpp

   Core Class implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu Jan 20 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   CoreClass implementation
*/

#include <falcon/cclass.h>
#include <falcon/coreobject.h>
#include <falcon/vm.h>

namespace Falcon {

CoreClass::CoreClass( const Symbol *sym, LiveModule *lmod, PropertyTable *pt ):
   Garbageable(),
   m_lmod( lmod ),
   m_sym( sym ),
   m_properties( pt ),
   m_factory(sym->getClassDef()->factory())
{
}

bool CoreClass::derivedFrom( const String &className ) const
{
   // else search the base class name in the inheritance properties.
   uint32 pos;
   if ( m_properties->findKey( className, pos ) )
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


void CoreClass::gcMark( uint32 gen ) const
{
   // first, mark ourselves.
   mark( gen );

   // then mark our items,
   memPool->markItem( m_constructor );
   for( uint32 i = 0; i < properties().added(); i++ )
   {
      // ancestors are in the property table as classItems
      memPool->markItem( *properties().getValue(i) );
   }

   // and our module
   m_lmod->mark( gen );
}


CoreObject *CoreClass::createInstance( void *userdata, bool bDeserial ) const
{
   if ( m_sym->isEnum() )
   {
      throw new CodeError( ErrorParam( e_noninst_cls, __LINE__ )
            .extra( m_sym->name() ) );
      // anyhow, flow through to allow user to see the object
   }


   // The core object will self configure,
   // eventually calling the user data constructor and creating the property vector.
   CoreObject *instance = m_factory( this, userdata, bDeserial );

   return instance;
}

}


/* end of coreclass.cpp */
