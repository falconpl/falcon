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
#include <falcon/itemdict.h>

namespace Falcon {

CoreClass::CoreClass( const Symbol *sym, LiveModule *lmod, PropertyTable *pt ):
   Garbageable(),
   m_lmod( lmod ),
   m_sym( sym ),
   m_properties( pt ),
   m_factory(sym->getClassDef()->factory()),
   m_states( 0 ),
   m_initState( 0 ),
   m_bHasInitEnter( false )
{
}

bool CoreClass::derivedFrom( const String &className ) const
{
   // is this class?
   if ( m_sym->name() == className )
      return true;

   // else, try with the properties/inheritance
   for( uint32 i = 0; i < properties().added(); ++i )
   {
      const Item& p = *properties().getValue(i)->dereference();
      if( p.isClass() && p.asClass()->derivedFrom( className ) )
      {
         return true;
      }
   }

   return false;
}

bool CoreClass::derivedFrom( const Symbol *sym ) const
{
   // is this class?
   /*if ( m_sym == sym || m_sym->name() == sym->name() )
      return true;
   */

   if ( m_sym == sym )
         return true;

   // else, try with the properties/inheritance
   for( uint32 i = 0; i < properties().added(); ++i )
   {
      const Item& p = *properties().getValue(i)->dereference();
      if( p.isClass() && p.asClass()->derivedFrom( sym ) )
      {
         return true;
      }
   }

   return false;
}



CoreClass::~CoreClass()
{
   delete m_properties;
   delete m_states;
}


void CoreClass::gcMark( uint32 gen )
{
   // first, mark ourselves.
   if ( gen != mark() )
   {
      mark( gen );

      // then mark our items,
      memPool->markItem( m_constructor );
      for( uint32 i = 0; i < properties().added(); i++ )
      {
         // ancestors are in the property table as classItems
         memPool->markItem( *properties().getValue(i) );
      }

      // and our states
      if( m_states != 0 )
      {
         m_states->gcMark( gen );
      }

      // and our module
      m_lmod->gcMark( gen );
   }
}


void CoreClass::states( ItemDict* sd, ItemDict* is )
{
   delete m_states;
   m_states = sd;
   // have we got an init state?
   m_initState = is;

   String name("__enter");
   if( is != 0 && is->find( &name ) )
      m_bHasInitEnter = true;

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

   if( m_initState != 0 )
   {
      instance->setState( "init", m_initState );
   }

   return instance;
}

}


/* end of coreclass.cpp */
