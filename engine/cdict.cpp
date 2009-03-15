/*
   FALCON - The Falcon Programming Language.
   FILE: cdict.cpp

   Core dictionary common functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 04 Jan 2009 09:49:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/cdict.h>
#include <falcon/vm.h>

namespace Falcon {

Item *CoreDict::find( const String &key ) const
{
   return find( const_cast<String *>(&key) );
}


void CoreDict::readProperty( const String &prop, Item &item )
{
   if( m_blessed )
   {
      Item *method;
      
      if ( ( method = find( prop ) ) != 0 )
      {
         method = method->dereference();
         if ( method->isFunction() )
         {
            method->setMethod( this, method->asFunction() );
         }

         item = *method;
         return;
      }
   }

   // try to find a generic method
   VMachine *vm = VMachine::getCurrent();
   if( vm != 0 )
   {
      CoreClass* cc = vm->getMetaClass( FLC_ITEM_DICT );
      uint32 id;
      if ( cc == 0 || ! cc->properties().findKey( prop, id ) )
      {
         throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
      }
      item = *cc->properties().getValue( id );
      item.methodize( this );
   }

}


void CoreDict::writeProperty( const String &prop, const Item &item )
{
   if( m_blessed )
   {
      Item *method;
      if ( ( method = find( prop ) ) != 0 )
      {
         *method = *item.dereference();
         return;
      }
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
}


void CoreDict::readIndex( const Item &pos, Item &target )
{
   if( ! find( *pos.dereference(), target ) )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ) );
   }
}

void CoreDict::writeIndex( const Item &pos, const Item &target )
{
   /*
   if( m_blessed )
   {
      Item *method;
      String index( "setIndex__" );
      if ( (method = find( &index ) ) != 0 )
      {
         method = method->dereference();
         if ( method->isFunction() )
         {
            VMachine* vm = origin();
            method->setTabMethod( this, method->asFunction(), method->asModule() );
            vm->pushParameter( pos );
            vm->pushParameter( target );
            vm->callItemAtomic( *method, 2 );
            return;
         }
      }
   }
   */

   const Item *tgt = target.dereference();
   if( tgt->isString() )
   {
      //TODO: Methodize
      insert( *pos.dereference(), new CoreString( *tgt->asString() ) );
   }
   else {
      insert( *pos.dereference(), *tgt );
   }

}

}

/* end of cdict.cpp */
