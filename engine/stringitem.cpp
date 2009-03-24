/*
   FALCON - The Falcon Programming Language.
   FILE: string.cpp

   Item-oriented Core String implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 03 Jan 2009 23:43:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/error.h>
#include <falcon/vm.h>
#include <stdio.h>

namespace Falcon
{

Item::Item( const String &str )
{
   CoreString* cstr = new CoreString( str );
   cstr->bufferize();
   setString( cstr );
}


StringGarbage::~StringGarbage()
{
}

bool StringGarbage::finalize()
{
   delete m_str;
   // as we're in m_str, we are destroyed here.

   return true; // prevent destructor to be called.
}

void String::readProperty( const String &prop, Item &item )
{
   VMachine *vm = VMachine::getCurrent();
   fassert( vm != 0 );

   // try to find a generic method
   CoreClass* cc = vm->getMetaClass( FLC_ITEM_STRING );
   if ( cc != 0 )
   {
      uint32 id;
      if( cc->properties().findKey( prop, id ) )
      {
         item = *cc->properties().getValue( id );
         item.methodize( this );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
}


void String::readIndex( const Item &index, Item &target )
{

   switch( index.type() )
   {
      case FLC_ITEM_INT:
         {
            int32 pos = (int32) index.asInteger();
            if ( checkPosBound( pos ) )
            {
               CoreString *gcs = new CoreString();
               gcs->append( getCharAt( pos ) );
               target = gcs;
               return;
            }
         }
         break;

      case FLC_ITEM_NUM:
      {
         int32 pos = (int32) index.asNumeric();
         if ( checkPosBound( pos ) )
         {
            CoreString *gcs = new CoreString();
            gcs->append( getCharAt( pos ) );
            target = gcs;
            return;
         }
      }
      break;

      case FLC_ITEM_RANGE:
      {
         int32 rstart = (int32) index.asRangeStart();
         if ( index.asRangeIsOpen() )
         {
            if ( checkPosBound( rstart ) ) {
               target = new CoreString( *this, rstart );
            }
            else {
               target = new CoreString;
            }
            return;
         }
         else {
            int32 rend = (int32) index.asRangeEnd();
            if ( checkRangeBound( rstart, rend ) )
            {
               target = new CoreString( *this, rstart, rend );
               return;
            }
            else {
               target = new CoreString;
            }
            return;
         }
      }
      break;

      case FLC_ITEM_REFERENCE:
         readIndex( index.asReference()->origin(), target );
         return;

   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "string" ) );
}


void String::writeIndex( const Item &index, const Item &target )
{

   register int32 pos;

   switch( index.type() )
   {
      case FLC_ITEM_INT:
         pos = (int32) index.asInteger();
         break;

      case FLC_ITEM_NUM:
         pos = (int32) index.asNumeric();
         break;

      case FLC_ITEM_RANGE:
      {
         if ( target.isString() )
         {
            register int pos = (int) index.asRangeStart();
            register int end = (int) (index.asRangeIsOpen() ? this->length() :  index.asRangeEnd());
            if ( checkRangeBound( pos, end ) )
            {
               if ( change( pos, end, *target.asString() ) )
               {
                  return;
               }
            }
         }
      }
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "string" ) );

      case FLC_ITEM_REFERENCE:
         writeIndex( index.asReference()->origin(), target );
         return;

      default:
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "string" ) );
   }


   if ( target.type() == FLC_ITEM_STRING )
   {
      String *cs_orig = target.asString();
      if( cs_orig->length() > 0 ) {
         if ( checkPosBound( pos ) ) {
            setCharAt( pos, cs_orig->getCharAt(0) );
            return;
         }
      }
   }
   else if( target.isOrdinal() )
   {
      int64 chr = target.forceInteger();
      if ( (chr >= 0) && (chr <= (int64) 0xFFFFFFFF) )
      {
         if ( checkPosBound( pos ) ) {
            setCharAt( pos, (uint32) chr );
            return;
         }
      }
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "string" ) );
}



}

/* end of stringitem.cpp */
