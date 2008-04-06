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
#include <falcon/cobject.h>
#include <falcon/vm.h>
#include <falcon/attribute.h>

namespace Falcon {

CoreClass::CoreClass( VMachine *origin, Symbol *sym, LiveModule *lmod, PropertyTable *pt ):
   m_lmod( lmod ),
   m_sym( sym ),
   m_properties( pt ),
   m_attributes( 0 ),
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
   setAttributeList( 0 );
}


CoreObject *CoreClass::createInstance( bool applyAttributes ) const
{
   if ( m_sym->isEnum() )
   {
      origin()->raiseError( new CodeError( ErrorParam( e_noninst_cls, __LINE__ ).
         extra( m_sym->name() ) ) );
      // anyhow, flow through to allow user to see the object
   }

   // we must have an origin pointer.
   CoreObject *instance = new CoreObject( origin(), *m_properties, symbol() );

   // assign attributes to the instance.
   if ( applyAttributes )
   {
      AttribHandler *head = m_attributes;
      while( head != 0 )
      {
         head->attrib()->giveTo( instance );
         head = head->next();
      }
   }

   return instance;
}


void CoreClass::addAttribute( Attribute *attrib )
{
   m_attributes->prev( new AttribHandler( attrib, 0, 0, m_attributes ) );
   m_attributes = m_attributes->prev();
}


void CoreClass::removeAttribute( Attribute *attrib )
{
   AttribHandler *head = m_attributes;
   if ( head == 0 )
      return;

   if ( head->attrib() == attrib )
   {
      m_attributes = m_attributes->next();
      m_attributes->prev( 0 );
      delete head;
      return;
   }

   head = head->next();
   while( head != 0 )
   {
      if ( head->attrib() == attrib )
      {
         head->prev()->next( head->next() );
         if ( head->next() != 0 )
            head->next()->prev( head->prev() );
         delete head;
         return;
      }

      head = head->next();
   }

}


void CoreClass::setAttributeList( AttribHandler *lst )
{
   while ( m_attributes != 0 )
   {
      AttribHandler *old = m_attributes;
      m_attributes = m_attributes->next();
      delete old;
   }

   m_attributes = lst;
}


}


/* end of coreclass.cpp */
