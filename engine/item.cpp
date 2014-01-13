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

#undef SRC
#define SRC "engine/item.cpp"

#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/stdhandlers.h>
#include <falcon/symbol.h>

#include <falcon/itemarray.h>
#include <falcon/itemdict.h>


namespace Falcon
{

Class* Item::m_funcClass;
Class* Item::m_stringClass;
Class* Item::m_dictClass;
Class* Item::m_arrayClass;

void Item::init(Engine* engine)
{
   m_funcClass = engine->handlers()->functionClass();
   m_stringClass = engine->handlers()->stringClass();
   m_dictClass = engine->handlers()->dictClass();
   m_arrayClass = engine->handlers()->arrayClass();
}

Item& Item::setString( const char* str )
{
   setUser( FALCON_GC_HANDLE(new String(str)) );
   return *this;
}

Item& Item::setString( const wchar_t* str )
{
   setUser( FALCON_GC_HANDLE(new String(str)) );
   return *this;
}

Item& Item::setString( const String& str )
{
   setUser( FALCON_GC_HANDLE(new String(str)) );
   return *this;
}


Item& Item::setSymbol( Symbol* sym )
{
   sym->incref();
   setUser( FALCON_GC_HANDLE(sym) );
   return *this;
}

//===========================================================================
// Generic item manipulators

bool Item::isCallable() const
{
   return isFunction()
            || (isArray() && (!asArray()->at(0).isArray() && asArray()->at(0).isCallable()))
            || isMethod()
            || isClass()
            || (isUser() && asClass()->typeID() == FLC_CLASS_ID_CLOSURE);
}

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

   case FLC_CLASS_ID_ARRAY:
      return ! asArray()->empty();

   case FLC_CLASS_ID_DICT:
      return ! asDict()->empty();

   case FLC_CLASS_ID_STRING:
      return ! asString()->empty();

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
         target.writeNumber( this->asNumeric(), FALCON_DEFAULT_NUMERIC_STRING_FORMAT );
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

         Engine::handlers()->functionClass()->describe( asMethodFunction(), temp, maxDepth-1, maxLength );
         target += temp;
         target += ")";
      }
      break;
      
      default:
      {
         Class* cls = 0;
         void* inst = 0;
         forceClassInst( cls, inst );
         cls->describe( inst, target, maxDepth, maxLength );
      }
      break;
   }
}


bool Item::clone( Item& target ) const
{
   void* data;

   if( type() >= FLC_ITEM_USER )
   {
     data = asClass()->clone( asInst() );
     if ( data == 0 )
        return false;
     
     target.setUser( asClass(), data );
   }
   else {
     target.copy( target );
   }

   return true;
}

int Item::compare( const Item& other ) const
{
   register const Item* i1 = this;
   register const Item* i2 = &other;
   
   int typeDiff = i1->type() - i2->type();
   if( typeDiff == 0 )
   {
      switch( i1->type() ) {
      case FLC_ITEM_NIL: return 0;
      case FLC_ITEM_INT: return (int) (i1->asInteger() - i2->asInteger());
      case FLC_ITEM_NUM: return (int) (i1->asNumeric() - i2->asNumeric());
      case FLC_ITEM_BOOL:
         if( i1->isTrue() )
         {
            if ( i2->isTrue() ) return 0;
            return 1;
         }
         else
         {
            if( i2->isTrue() ) return -1;
            return 0;
         }

      case FLC_CLASS_ID_STRING:
         return i1->asString()->compare(*i2->asString());

      default:
         return (int64) (static_cast<byte*>(i1->asInst()) - static_cast<byte*>(i2->asInst()));
      }
   }

   return typeDiff;
}



bool Item::exactlyEqual( const Item& other ) const
{
   register const Item* i1 = this;
   register const Item* i2 = &other;

   if( i1->type() != i2->type() )
   {
      if( i1->type() == FLC_ITEM_INT && i2->type() == FLC_ITEM_NUM )
      {
         return ((numeric) i1->asInteger()) == i2->asNumeric();
      }
      else if( i1->type() == FLC_ITEM_NUM && i2->type() == FLC_ITEM_INT )
      {
         return  i1->asNumeric() == ((numeric)i2->asInteger());
      }
      return false;
   }

   switch( i1->type() ) {
   case FLC_ITEM_NIL: return true;
   case FLC_ITEM_INT: return i1->asInteger() == i2->asInteger();
   case FLC_ITEM_NUM: return i1->asNumeric() == i2->asNumeric();
   case FLC_ITEM_BOOL:
      return i1->asBoolean() == i2->asBoolean();

   case FLC_CLASS_ID_STRING:
      return *i1->asString() == *i2->asString();

   default:
      return i1->asInst() == i2->asInst();
   }

   return false;
}


void* Item::asParentInst( Class* parent )
{
   Class* cls = 0;
   void* data = 0;
   if( asClassInst(cls, data ) )
   {
      return cls->getParentData(parent, data);
   }

   return 0;
}


void* Item::forceParentInst( Class* parent )
{
   Class* cls = 0;
   void* data = 0;
   forceClassInst(cls, data );
   return cls->getParentData(parent, data);
}

}

/* end of item.cpp */
