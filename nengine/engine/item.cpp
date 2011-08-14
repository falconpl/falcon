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
   setUser( (new String(str))->garbage() );
}

void Item::setString( const wchar_t* str )
{
   setUser( (new String(str))->garbage() );
}

void Item::setString( const String& str )
{
   setUser( (new String(str))->garbage() );
}

void Item::setString( String* str, bool bGarbage )
{
   static Class* strClass = Engine::instance()->stringClass();
   static Collector* coll = Engine::instance()->collector();
   if( bGarbage )
   {
      setUser(FALCON_GC_STORE(coll, strClass, str));
   }
   else
   {
      setUser( strClass, str );
   }
}

void Item::setArray( ItemArray* array, bool bGarbage )
{
   static Class* arrayClass = Engine::instance()->arrayClass();
   static Collector* coll = Engine::instance()->collector();

   if( bGarbage )
   {
      setUser( FALCON_GC_STORE(coll, arrayClass, array) );
   }
   else
   {
      setUser(arrayClass, array);
   }
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


void Item::describe( String &target, int maxDepth, int maxLength ) const
{
   target.size(0);

   switch( this->type() )
   {
      case FLC_ITEM_NIL:
         if ( isContinue() )
         {
            target = "continue";
         }
         else if ( isBreak() )
         {
            target = "break";
         }
         else
         {
            target = "Nil";
         }
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
         Engine::instance()->functionClass()->describe( asFunction(), target, maxDepth, maxLength );
      }
      break;

      case FLC_ITEM_METHOD:
      {         
         String temp;
         target = "(Method ";

         Item old = *this;
         old.unmethodize();
         old.describe( temp, 0, maxLength );
         target += temp + ".";
         temp = "";

         Engine::instance()->functionClass()->describe( asMethodFunction(), temp, maxDepth-1, maxLength );
         target += temp;
         target += ")";
      }
      break;

      case FLC_ITEM_REF:
      {
         asReference()->describe( target, maxDepth, maxLength );
         target = "Ref {" + target + "}";
      }
      break;

      
      case FLC_ITEM_USER:
      {
         asClass()->describe( asInst(), target, maxDepth, maxLength );
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
   case FLC_ITEM_USER:
     data = asClass()->clone( asInst() );
     if ( data == 0 )
        return false;
     target.setUser( asClass(), data );
     // we suppose that the other instance is freshly allocated,
     // -- if not, the class will know how to handle useless gcMarks
     target.garbage();
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
