/*
   FALCON - The Falcon Programming Language.
   FILE: coreobject.cpp

   Core object implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Jan 2009 19:46:05 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core object implementation.
*/

#include <falcon/vm.h>
#include <falcon/item.h>
#include <falcon/coreobject.h>
#include <falcon/symbol.h>
#include <falcon/cclass.h>
#include <falcon/stream.h>
#include <falcon/falcondata.h>

namespace Falcon
{

CoreObject::CoreObject( const CoreClass *parent ):
   Garbageable(),
   m_user_data( 0 ),
   m_bIsFalconData( false ),
   m_bIsSequence( false ),
   m_generatedBy( parent )
{
}

CoreObject::CoreObject( const CoreObject &other ):
   Garbageable( other ),
   m_user_data( 0 ),
   m_bIsFalconData( other.m_bIsFalconData ),
   m_bIsSequence( other.m_bIsFalconData ),
   m_generatedBy( other.m_generatedBy )
{
   if ( m_bIsFalconData )
   {
      fassert( other.m_user_data != 0 );
      m_user_data = other.getFalconData()->clone();
   }
   else {
      fassert( other.m_user_data == 0 );
   }
}


CoreObject::~CoreObject()
{
   if ( m_bIsFalconData )
      delete static_cast<FalconData *>( m_user_data );
}


void CoreObject::gcMark( uint32 gen )
{
   if( gen != mark() )
   {
      // mark ourseleves
      mark( gen );

      // our class
      const_cast<CoreClass*>(m_generatedBy)->gcMark( gen );

      // and possibly our inner falcon data
      if ( m_bIsFalconData )
      {
         fassert( m_user_data != 0 );
         static_cast<FalconData* >(m_user_data)->gcMark( gen );
      }
   }
}


bool CoreObject::serialize( Stream *stream, bool bLive ) const
{
   if( bLive )
   {
      stream->write( (byte *) &m_user_data, sizeof( m_user_data ) );
      return true;
   }
   return false;
}


bool CoreObject::hasProperty( const String &key ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   return pt.findKey( key, pos );
}


bool CoreObject::defaultProperty( const String &key, Item &prop ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   if ( pt.findKey( key, pos ) )
   {
      prop = *pt.getValue(pos);
      prop.methodize( const_cast<CoreObject *>(this) );
      return true;
   }

   return false;
}


void CoreObject::readOnlyError( const String &key ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   throw new AccessError( ErrorParam( pt.findKey( key, pos ) ? e_prop_ro : e_prop_acc, __LINE__ )
         .extra( key ) );
}

bool CoreObject::deserialize( Stream *stream, bool bLive )
{
   if( bLive )
   {
      if ( stream->read( (byte *) &m_user_data, sizeof( m_user_data ) ) != sizeof( m_user_data ) )
         return false;
      return true;
   }

   return false;
}



bool CoreObject::derivedFrom( const String &className ) const
{
   const Symbol *clssym = m_generatedBy->symbol();
   return (clssym->name() == className || m_generatedBy->derivedFrom( className ));
}


bool CoreObject::getMethodDefault( const String &name, Item &mth ) const
{
   const Falcon::Item* pmth = generator()->properties().getValue( name );

   if ( pmth != 0 && pmth->isFunction() )
   {
      // yes, a valid method
      mth = *pmth;

      mth.methodize( SafeItem( const_cast<CoreObject*>(this) ) );
      return true;
   }
   return false;
}


//=======================================================================
// Deep item overloading
//=======================================================================


bool CoreObject::setProperty( const String &propName, const String &value )
{
   return setProperty( propName, new CoreString( value ) );
}

void CoreObject::readIndex( const Item &pos, Item &target )
{
   if ( getMethod( "getIndex__", target ) )
   {
      VMachine* vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParameter( pos );
         vm->callItemAtomic( target, 1 );
         target = vm->regA();
         return;
      }
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "getIndex__" ) );
}

void CoreObject::writeIndex( const Item &pos, const Item &target )
{
   Item method;
   if ( getMethod( "setIndex__", method ) )
   {
      VMachine* vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParameter( pos );
         vm->pushParameter( target );
         vm->callItemAtomic( method, 2 );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "setIndex__" ) );
}

void CoreObject::readProperty( const String &prop, Item &target )
{
   Item *p;

   if ( ! getProperty( prop, target ) )
   {
      // try to find a generic method
      VMachine* vm = VMachine::getCurrent();
      fassert( vm != 0 );
      CoreClass* cc = vm->getMetaClass( FLC_ITEM_OBJECT );
      uint32 id;
      if ( cc == 0 || ! cc->properties().findKey( prop, id ) )
      {
         throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
      }

      p = cc->properties().getValue( id );
   }
   else
      p = target.dereference();

   switch( p->type() ) {

      case FLC_ITEM_CLASS:
         if ( derivedFrom( p->asClass()->symbol()->name() ) )
            target.setClassMethod( this, p->asClass() );
         else
            target.setClass( p->asClass() );
      break;

      default:
        target = *p;
        target.methodize( this );
   }
}

void CoreObject::writeProperty( const String &prop, const Item &target )
{
   if ( ! setProperty( prop, target ) )
   {
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
   }
}


}

/* end of coreobject.cpp */
