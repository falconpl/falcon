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
#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/common.h>
#include <falcon/symbol.h>
#include <falcon/cobject.h>
#include <falcon/carray.h>
#include <falcon/garbagepointer.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/membuf.h>
#include <falcon/attribute.h>
#include <falcon/vmmaps.h>
#include <cstdlib>
#include <cstring>

namespace Falcon {

void Item::setGCPointer( VMachine *vm, FalconData *ptr, uint32 sig )
{
   type( FLC_ITEM_POINTER );
   m_data.gptr.signature = sig;
   m_data.gptr.gcptr = new GarbagePointer( vm, ptr );
}

void Item::setGCPointer( GarbagePointer *shell, uint32 sig )
{
   type( FLC_ITEM_POINTER );
   m_data.gptr.signature = sig;
   m_data.gptr.gcptr = shell;
}

FalconData *Item::asGCPointer() const
{
   return m_data.gptr.gcptr->ptr();
}


// This function is called solely if other and this have the same type.
bool Item::internal_is_equal( const Item &other ) const
{
   switch( type() )
   {
      case FLC_ITEM_NIL: return true;

      case FLC_ITEM_BOOL:
         return asBoolean() == other.asBoolean();

      case FLC_ITEM_RANGE:
         return asRangeStart() == other.asRangeStart() && asRangeEnd() == other.asRangeEnd() &&
               asRangeStep() == other.asRangeStep() &&
               asRangeIsOpen() == other.asRangeIsOpen();

      case FLC_ITEM_INT:
         return asInteger() == other.asInteger();

      case FLC_ITEM_NUM:
         return asNumeric() == other.asNumeric();

      case FLC_ITEM_ATTRIBUTE:
         return asAttribute() == other.asAttribute();

      case FLC_ITEM_STRING:
         return *asString() == *other.asString();

      case FLC_ITEM_LBIND:
         return asLBind() != 0 && other.asLBind() != 0 && *asLBind() == *other.asLBind();

      case FLC_ITEM_ARRAY:
         // for now, just compare the pointers.
         return asArray() == other.asArray();

      case FLC_ITEM_DICT:
         return asDict() == other.asDict();

      case FLC_ITEM_FUNC:
         return asFunction() == other.asFunction();

      case FLC_ITEM_OBJECT:
         return asObject() == other.asObject();

      case FLC_ITEM_MEMBUF:
         return asMemBuf() == other.asMemBuf();

      case FLC_ITEM_METHOD:
         return asMethodObject() == other.asMethodObject() && asMethodFunction() == other.asMethodFunction();

      case FLC_ITEM_REFERENCE:
         return asReference() == other.asReference();

      case FLC_ITEM_FBOM:
         if ( getFbomMethod() == other.getFbomMethod() ) {
            Item lt, lo;
            getFbomItem( lt );
            other.getFbomItem( lo );
            return lt == lo;
         }
         return false;

      case FLC_ITEM_CLASS:
         return asClass()->symbol() == other.asClass()->symbol();
   }

   return false;
}


bool Item::isTrue() const
{
   switch( dereference()->type() )
   {
      case FLC_ITEM_BOOL:
         return asBoolean() != 0;

      case FLC_ITEM_INT:
         return asInteger() != 0;

      case FLC_ITEM_NUM:
         return asNumeric() != 0.0;

      case FLC_ITEM_RANGE:
         return asRangeStart() != asRangeEnd() || asRangeIsOpen();

      case FLC_ITEM_STRING:
         return asString()->size() != 0;

      case FLC_ITEM_ARRAY:
         return asArray()->length() != 0;

      case FLC_ITEM_DICT:
         return asDict()->length() != 0;

      case FLC_ITEM_FUNC:
      case FLC_ITEM_OBJECT:
      case FLC_ITEM_CLASS:
      case FLC_ITEM_METHOD:
      case FLC_ITEM_FBOM:
      case FLC_ITEM_ATTRIBUTE:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_LBIND:
         // methods are always filled, so they are always true.
         return true;
   }

   return false;
}

static int64 s_atoi( const String *cs )
{
   if ( cs->size() == 0 )
      return 0;
   const char *p =  (const char *)cs->getRawStorage() + ( cs->size() -1 );
   uint64 val = 0;
   uint64 base = 1;
   while( p > (const char *)cs->getRawStorage() ) {
      if ( *p < '0' || *p > '9' ) {
         return 0;
      }
      val += (*p-'0') * base;
      p--;
      base *= 10;
   }
   if ( *p == '-' ) return -(int64)val;
   return (int64)(val*base);
}

int64 Item::forceInteger() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return asInteger();

      case FLC_ITEM_NUM:
         return (int64) asNumeric();

      case FLC_ITEM_RANGE:
         return (int64)asRangeStart();

      case FLC_ITEM_STRING:
      {
         int64 tgt;
         if ( asString()->parseInt( tgt ) )
            return tgt;
         return 0;
      }
   }
   return 0;
}


numeric Item::forceNumeric() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return (numeric) asInteger();

      case FLC_ITEM_NUM:
         return asNumeric();

      case FLC_ITEM_RANGE:
         return (numeric) asRangeStart();

      case FLC_ITEM_STRING:
      {
         double tgt;
         if ( asString()->parseDouble( tgt ) )
            return tgt;
         return 0.0;
      }
   }
   return 0.0;
}


int Item::compare( const Item &other ) const
{
   switch ( type() << 8 | other.type() ) {
      case FLC_ITEM_NIL<<8 | FLC_ITEM_NIL:
         return 0;

      case FLC_ITEM_BOOL<<8 | FLC_ITEM_BOOL:
         if ( other.asBoolean() == asBoolean() )
            return 0;
         return ( asBoolean() > other.asBoolean() ) ? 1:-1;

      case FLC_ITEM_INT<<8 | FLC_ITEM_INT:
         if ( asInteger() < other.asInteger() ) return -1;
         else if ( asInteger() > other.asInteger() ) return 1;
         else return 0;

      case FLC_ITEM_INT<<8 | FLC_ITEM_NUM:
         if ( ((numeric)asInteger()) < other.asNumeric() ) return -1;
         else if ( ((numeric)asInteger()) > other.asNumeric() ) return 1;
         else return 0;

      case FLC_ITEM_NUM<<8 | FLC_ITEM_INT:
         if ( asNumeric() < other.asInteger() ) return -1;
         else if ( asNumeric() > other.asInteger() ) return 1;
         else return 0;

      case FLC_ITEM_NUM<<8 | FLC_ITEM_NUM:
         if ( asNumeric() < other.asNumeric() ) return -1;
         else if ( asNumeric() > other.asNumeric() ) return 1;
         else return 0;

      case FLC_ITEM_STRING << 8 | FLC_ITEM_STRING:
         return asString()->compare( *other.asString() );
   }

   if ( type() < other.type() ) return -1;
   if ( type() > other.type() ) return 1;
   return internal_compare( other );

}

int Item::internal_compare( const Item &other ) const
{
   switch ( type() )
   {
      case FLC_ITEM_NIL:
         return 0;

      case FLC_ITEM_BOOL:
         if ( other.asBoolean() == asBoolean() )
            return 0;
         return ( asBoolean() > other.asBoolean() ) ? 1:-1;

      case FLC_ITEM_INT:
         if ( asInteger() < other.asInteger() ) return -1;
         else if ( asInteger() > other.asInteger() ) return 1;
         else return 0;

      case FLC_ITEM_NUM:
         if ( asNumeric() < other.asNumeric() ) return -1;
         else if ( asNumeric() > other.asNumeric() ) return 1;
         else return 0;

      case FLC_ITEM_ATTRIBUTE:
         if( asAttribute() > other.asAttribute() )
            return 1;
         else if ( asAttribute() > other.asAttribute() )
            return -1;
         return 0;

      case FLC_ITEM_STRING:
         return asString()->compare( *other.asString() );

      case FLC_ITEM_LBIND:
         if ( asLBind() == 0 ) return -1;
         if ( other.asLBind() == 0 ) return 1;
         return asLBind()->compare( *other.asLBind() );

      case FLC_ITEM_RANGE:
         if ( asRangeStart() < other.asRangeStart() ) return -1;
         if ( asRangeStart() > other.asRangeStart() ) return 1;
         if ( asRangeIsOpen() )
         {
            if ( other.asRangeIsOpen() ) return 0;
            return 1;
         }

         if ( other.asRangeIsOpen() )
            return -1;

         if ( asRangeEnd() < other.asRangeEnd() ) return -1;
         if ( asRangeEnd() > other.asRangeEnd() ) return 1;
         if ( asRangeStep() < other.asRangeStep() ) return -1;
         if ( asRangeStep() > other.asRangeStep() ) return 1;
         return 0;

      case FLC_ITEM_ARRAY:
         if ( asArray() < other.asArray() ) return -1;
         else if ( asArray() > other.asArray() ) return 1;
         else return 0;

      case FLC_ITEM_DICT:
         if ( asDict() < other.asDict() ) return -1;
         else if ( asDict() > other.asDict() ) return 1;
         else return 0;

      case FLC_ITEM_FUNC:
         if ( asFunction() < other.asFunction() ) return -1;
         else if ( asFunction() > other.asFunction() ) return 1;
         else return 0;

      case FLC_ITEM_OBJECT:
         if ( asObject() < other.asObject() ) return -1;
         else if ( asObject() > other.asObject() ) return 1;
         else return 0;

      case FLC_ITEM_MEMBUF:
         if ( asMemBuf() < other.asMemBuf() ) return -1;
         else if ( asMemBuf() > other.asMemBuf() ) return 1;
         else return 0;

      case FLC_ITEM_CLSMETHOD:
      case FLC_ITEM_METHOD:
         if( asMethodObject() > other.asMethodObject() )
            return 1;
         if( asMethodObject() < other.asMethodObject() )
            return -1;
         if( asMethodFunction() > other.asMethodFunction() )
            return 1;
         if( asMethodFunction() < other.asMethodFunction() )
            return -1;
         return 0;

      case FLC_ITEM_FBOM:
         if ( getFbomMethod() == other.getFbomMethod() ) {
            Item lt, lo;
            getFbomItem( lt );
            other.getFbomItem( lo );
            return lt.compare( lo );
         }
         if( getFbomMethod() > other.getFbomMethod() )
            return 1;
         return -1;


      case FLC_ITEM_CLASS:
         if ( asClass() < other.asClass() ) return -1;
         else if ( asClass() > other.asClass() ) return 1;
         else return 0;

      // having a reference here means that some reference has been
      // injected in some object using compare() from C code. In example,
      // dictionary keys. So, we can't check the value of the dereference,
      // we must use comparation on this item as if this item was an opaque
      // type.
      case FLC_ITEM_REFERENCE:
         if ( asReference() < other.asReference() ) return -1;
         if ( asReference() > other.asReference() ) return 1;
         return 0;

   }

   return 0;
}

bool Item::isOfClass( const String &className ) const
{
   switch( type() )
   {
      case FLC_ITEM_OBJECT:
         // objects may be classless or derived from exactly one class.
         return asObject()->derivedFrom( className );

      case FLC_ITEM_CLASS:
         return className == asClass()->symbol()->name() || asClass()->derivedFrom( className );

      case FLC_ITEM_METHOD:
         return asMethodObject()->derivedFrom( className );
   }

   return false;
}


void Item::toString( String &target ) const
{
   bool saveHstr = false;
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

      case FLC_ITEM_RANGE:
         target = "[";
         target.writeNumber( (int64) this->asRangeStart() );
         target += ":";
         if ( ! this->asRangeIsOpen() )
         {
            target.writeNumber( (int64) this->asRangeEnd() );
            if ( this->asRangeStep() !=  0 )
            {
               target += ":";
               target.writeNumber( (int64) this->asRangeStep() );
            }
         }
         target += "]";
      break;

      case FLC_ITEM_NUM:
      {
         target.writeNumber( this->asNumeric(), "%.16g" );
      }
      break;

      case FLC_ITEM_MEMBUF:
         target = "{MemBuf of ";
         target.writeNumber( (int64) this->asMemBuf()->length() );
         target += " words long ";
            target.writeNumber( (int64) this->asMemBuf()->wordSize() );
         target += "bytes }";
      break;

      case FLC_ITEM_ATTRIBUTE:
         target = "{attrib:" + asAttribute()->name() + "}";
      break;

      case FLC_ITEM_STRING:
         target = *asString();
      break;

      case FLC_ITEM_LBIND:
         if ( asLBind() == 0 )
            target = "Nil";
         else
            target = "&" + *asLBind();
      break;

      case FLC_ITEM_REFERENCE:
         target = "{Ref to ";
         dereference()->toString( target );
         target += "}";
      break;

      case FLC_ITEM_OBJECT:
         target = "Object";
      break;

      case FLC_ITEM_ARRAY:
         target = "Array";
      break;

      case FLC_ITEM_DICT:
         target = "Dictionary";
      break;

      case FLC_ITEM_FUNC:
         target = "Function " + this->asFunction()->name();
      break;

      case FLC_ITEM_CLASS:
         target = "Class " + this->asClass()->symbol()->name();
      break;

      case FLC_ITEM_METHOD:
         target = "Method " + this->asMethodFunction()->name();
      break;


      case FLC_ITEM_CLSMETHOD:
         target = "ClsMeth " + this->asMethodClass()->symbol()->name();
      break;

      default:
         target = "<?>";
   }
}

bool Item::methodize( const CoreObject *self )
{
   Item *data = dereference();

   switch( data->type() )
   {
      case FLC_ITEM_FUNC:
      {
         data->setMethod( const_cast< CoreObject *>(self), data->asFunction(), data->asModule() );
      }
      return true;

      case FLC_ITEM_METHOD:
         data->setMethod( const_cast< CoreObject *>(self), data->asMethodFunction(), data->asModule() );
      return true;

      case FLC_ITEM_ARRAY:
         if ( data->asArray()->length() > 0 )
         {
            Item *citem = &data->asArray()->at(0);
            if ( citem->isMethod() && citem->asMethodObject() == self )
            {
               return true;
            }
            else if ( citem->isCallable() )
            {
               *data = data->asArray()->clone();
               data->asArray()->at(0).methodize( self );
               return true;
            }
         }
      return false;
   }

   return false;
}

void Item::destroy()
{
   switch( this->type() )
   {
      case FLC_ITEM_STRING:
         delete asString();
      break;

      case FLC_ITEM_MEMBUF:
         delete asMemBuf();
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *arr = asArray();
         for( uint32 i = 0; i < arr->length(); i++ )
            arr->elements()[i].destroy();
         delete arr;
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = asDict();
         Item key, value;
         dict->traverseBegin();
         while( dict->traverseNext( key, value ) )
         {
            key.destroy();
            value.destroy();
         }
         delete dict;
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *obj = asObject();
         for( uint32 i = 0; i < obj->propCount(); i++ )
         {
            Item tmp;
            obj->getPropertyAt(i, tmp);
            tmp.destroy();
         }
         delete obj;
      }
      break;

      case FLC_ITEM_CLASS:
      {
         CoreClass *cls = asClass();

         cls->constructor().destroy();
         PropertyTable &props = cls->properties();
         for( uint32 i = 0; i < props.size(); i++ ) {
            props.getValue(i)->destroy();
         }
         delete cls;
      }
      break;

      case FLC_ITEM_FBOM:
         {
            Item fbitm;
            getFbomItem( fbitm );
            fbitm.destroy();
         }
         break;
      case FLC_ITEM_METHOD:
         {
         Item(asMethodObject()).destroy();
         }
      break;

      case FLC_ITEM_CLSMETHOD:
      {
         Item(asMethodObject() ).destroy();
         Item(asMethodClass() ).destroy();
      }
      break;

      case FLC_ITEM_REFERENCE:
         asReference()->origin().destroy();
         delete asReference();
      break;
   }

   setNil();
}

bool Item::isCallable() const
{

   if ( isFbom() || isClassMethod() || isClass() )
      return true;

   // simple case: normally callable item
   if( isFunction() || isMethod() )
   {
      // Detached?
      if ( ! m_data.ptr.m_liveMod->isAlive() )
      {
         const_cast<Item *>(this)->setNil();
         return false;
      }
      return true;
   }

   //a bit more complex: a callable array...
   if( type() == FLC_ITEM_ARRAY )
   {
      CoreArray *arr = asArray();
      if ( arr->length() > 0 )
      {
         const Item &first = arr->at(0);
         if ( ! first.isArray() && first.isCallable() )
            return true;
      }
   }

   // in all the other cases, the item is not callable
   return false;
}

bool Item::isLBind() const
{
   if ( type() == FLC_ITEM_LBIND )
   {
      if ( m_data.ptr.m_liveMod->module() == 0 )
      {
         const_cast< Item*>(this)->setNil();
         return false;
      }

      return true;
   }

   return false;
}

const String *Item::asLBind() const
{
   if ( type() == FLC_ITEM_LBIND )
   {
      LiveModule *lmod = m_data.ptr.m_liveMod;
      if ( lmod->module() == 0 )
      {
         const_cast< Item*>(this)->setNil();
         return 0;
      }

      return lmod->module()->getString( (uint32) m_data.num.val1 );
   }

   return 0;
}

}

/* end of item.cpp */
