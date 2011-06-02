/*
   FALCON - The Falcon Programming Language.
   FILE: item.cpp

   Item API implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ott 12 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Item API implementation
   This File contains the non-inlined access to items.
*/

#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/engine.h>

namespace Falcon
{

void Item::setString( const char* str )
{
   setDeep( (new String(str))->garbage() );
}

void Item::setString( const wchar_t* str )
{
   setDeep( (new String(str))->garbage() );
}

void Item::setString( const String& str )
{
   setDeep( (new String(str))->garbage() );
}

void Item::setString( String* str )
{
   setDeep( str->garbage() );
}

void Item::setArray( ItemArray* array )
{
   Engine* eng = Engine::instance();
   setDeep( eng->collector()->store(eng->arrayClass(), array) );
}

/*
void Item::setDict( ItemDictionary* dict )
{
   setDeep( dict, Engine::instance()->dictClass() );
}
*/
//===========================================================================
// Generic item manipulators

bool Item::isTrue() const
{
   switch( type() )
   {
   case FLC_ITEM_NIL:
      return false;

   case FLC_ITEM_BOOL:
      return asBoolean() != 0;

   case FLC_ITEM_INT:
      return asInteger() != 0;

   case FLC_ITEM_NUM:
      return asNumeric() != 0.0;

   default:
      return false;
   }

   return false;
}

int64 Item::forceInteger() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return asInteger();

      case FLC_ITEM_NUM:
         return (int64) asNumeric();
   }
   return 0;
}


int64 Item::forceIntegerEx() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return asInteger();

      case FLC_ITEM_NUM:
         return (int64) asNumeric();

   }
   //throw new TypeError( ErrorParam( e_param_type, __LINE__ ) );

   // to make some dumb compiler happy
   return 0;
}


numeric Item::forceNumeric() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return (numeric) asInteger();

      case FLC_ITEM_NUM:
         return asNumeric();
   }
   return 0.0;
}


void Item::describe( String &target ) const
{
   target.size(0);

   switch( this->type() )
   {
      case FLC_ITEM_NIL:
         target = "Nil";
      break;

      case FLC_ITEM_BOOL:
         target = asBoolean() ? "true" : "false";
      break;


      case FLC_ITEM_INT:
         target.writeNumber( this->asInteger() );
      break;

      case FLC_ITEM_NUM:
      {
         target.writeNumber( this->asNumeric(), "%.16g" );
      }
      break;

      case FLC_ITEM_FUNC:
      {
         Engine::instance()->functionClass()->describe( asFunction(), target );
      }
      break;

      case FLC_ITEM_DEEP:
      {
         asDeepClass()->describe( asDeepInst(), target );
      }
      break;

      case FLC_ITEM_USER:
      {
         asUserClass()->describe( asUserInst(), target );
      }
      break;

      default:
         target = "<?>";
   }
}


bool Item::clone( Item& target ) const
{
   void* data;

   switch ( type() )
   {
   case FLC_ITEM_DEEP:
     data = asDeepClass()->clone( asDeepInst() );
     if ( data == 0 )
        return false;
     target.setDeep( content.data.pToken->collector()->store(asDeepClass(), data) );
     break;

   case FLC_ITEM_USER:
     data = asUserClass()->clone( asUserInst() );
     if ( data == 0 )
        return false;
     target.setUser( asUserClass(), data );
     break;

   default:
     target.copy( target );
   }

   return true;
}

int Item::compare( const Item& other ) const
{
   int typeDiff = type() - other.type();
   if( typeDiff == 0 )
   {
      switch( type() ) {
      case FLC_ITEM_NIL: return 0;
      case FLC_ITEM_INT: return (int) (asInteger() - other.asInteger());
      case FLC_ITEM_NUM: return (int) (asNumeric() - other.asNumeric());
      case FLC_ITEM_BOOL:
         if( isTrue() )
         {
            if ( other.isTrue() ) return 0;
            return 1;
         }
         else
         {
            if( other.isTrue() ) return -1;
            return 0;
         }

      default:
         return (int64) (this - &other);
      }
   }

   return typeDiff;
}

}

/* end of item.cpp */
