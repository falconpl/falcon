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


#include <falcon/coredict.h>
#include <falcon/vm.h>

namespace Falcon {

bool CoreDict::getMethod( const String &name, Item &mth )
{
   if ( m_blessed )
   {
      Item* found = find( name );
      if ( found != 0 )
      {
         mth = *found;
         return mth.methodize( SafeItem( this ) );
      }
   }

   return false;
}



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
         item = *method->dereference();
         item.methodize( this );  // may fail but it's ok
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
   if( m_blessed )
   {
      Item *method;
      if ( (method = find( OVERRIDE_OP_GETINDEX ) ) != 0 )
      {
         Item mth = *method;
         if ( mth.methodize(this) )
         {
            VMachine* vm = VMachine::getCurrent();
            if( vm != 0 )
            {
               vm->pushParam( pos );
               vm->callItemAtomic( mth, 1 );
            }
            return;
         }
      }
   }

   if( ! find( *pos.dereference(), target ) )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ) );
   }
}

void CoreDict::writeIndex( const Item &pos, const Item &target )
{
   if( m_blessed )
   {
      Item *method;
      if ( (method = find( OVERRIDE_OP_SETINDEX ) ) != 0 )
      {
         Item mth = *method;
         if ( mth.methodize(this) )
         {
            VMachine* vm = VMachine::getCurrent();
            if( vm != 0 )
            {
               vm->pushParam( pos );
               vm->pushParam( target );
               vm->callItemAtomic( mth, 2 );
               return;
            }
         }
      }
   }

   const Item *tgt = target.dereference();
   if( tgt->isString() )
   {
      //TODO: Methodize
      put( *pos.dereference(), new CoreString( *tgt->asString() ) );
   }
   else {
      put( *pos.dereference(), *tgt );
   }
}

void CoreDict::gcMark( uint32 gen )
{
   if ( gen != mark() )
   {
      mark( gen );
      m_dict->gcMark( gen );
   }
}

bool CoreDict::find( const Item &key, Item &value )
{
   Item *itm;
   if( ( itm = find( key ) ) != 0 )
   {
     value = *itm;
     return true;
   }
   return false;
}


//TODO - move in another file
int ItemDict::compare( const ItemDict& other, ItemDict::Parentship* parent ) const
{
   // really the same.
   if (&other == this)
      return 0;
      
   // Create the new parentship
   Parentship current( this, parent );
   
   Iterator ithis( const_cast<ItemDict*>(this) );
   Iterator iother( const_cast<ItemDict*>(&other) );
   
   while( ithis.hasCurrent() )
   {
      // is the other shorter?
      if ( ! iother.hasCurrent() )
      {
         // we're bigger
         return 1;
      }
      
      const Item& tkey = ithis.getCurrentKey();
      const Item& okey = iother.getCurrentKey();
      int v = checkValue( tkey, okey, current );
      if ( v != 0 )
         return v;
      
      const Item& tvalue = ithis.getCurrent();
      const Item& ovalue = iother.getCurrent();
      v = checkValue( tvalue, ovalue, current );
      if ( v != 0 )
         return v;
      
      ithis.next();
      iother.next();
   }
   
   if( iother.hasCurrent() )
      return -1;
   
   //  ok, we're the same
   return 0;
}


int ItemDict::checkValue( const Item& first, const Item& second, ItemDict::Parentship& current ) const
{
   // different dictionaries?
   if ( first.isDict() && first.isDict() )
   {
      const ItemDict* dict = &first.asDict()->items();
      Parentship *p1 = current.m_parent;
      // If it is not, we should scan it too.
      bool bDescend = true;
      
      while( p1 != 0 )
      {
         if( p1->m_dict == dict )
         {
            bDescend = false;
            break;
         }
         p1 = p1->m_parent;
      }
      
      if ( bDescend )
      {
         return dict->compare( second.asDict()->items(), &current );
      }
      
      return 0;
   }

   return first.compare( second );
}

}

/* end of cdict.cpp */
