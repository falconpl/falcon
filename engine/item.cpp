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
#include <falcon/coreobject.h>
#include <falcon/corefunc.h>
#include <falcon/carray.h>
#include <falcon/garbagepointer.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/membuf.h>
#include <falcon/vmmaps.h>
#include <falcon/error.h>
#include <cstdlib>
#include <cstring>


namespace Falcon
{

inline void assignToVm( Garbageable *gcdata )
{
   gcdata->mark( memPool->generation() );
}

//=================================================================
// Garbage markers
Item::Item( byte t, Garbageable *dt )
{
   type( t );
   content( dt );
   assignToVm( dt );
}

void Item::setRange( CoreRange *r )
{
   type( FLC_ITEM_RANGE );
   all.ctx.data.content = r;
   assignToVm( r );
}


void Item::setString( String *str )
{
   type( FLC_ITEM_STRING );

   all.ctx.data.ptr.voidp = str;
   all.ctx.data.ptr.extra = 0;
   if ( str->isCore() )
      assignToVm( &static_cast<CoreString*>(str)->garbage() );
}

void Item::setString( String *str, LiveModule *lm )
{
   type( FLC_ITEM_STRING );
   // must not be a non-garbage string
   fassert( ! str->isCore() );
   all.ctx.data.ptr.voidp = str;
   all.ctx.data.ptr.extra = lm;
}

void Item::setArray( CoreArray *array )
{
   type( FLC_ITEM_ARRAY );
   all.ctx.data.ptr.voidp = array;
   assignToVm( array );
}

void Item::setObject( CoreObject *obj )
{
   type( FLC_ITEM_OBJECT );
   all.ctx.data.ptr.voidp = obj;
   assignToVm( obj );
}


void Item::setDict( CoreDict *dict )
{
   type( FLC_ITEM_DICT );
   all.ctx.data.ptr.voidp = dict;
   assignToVm( dict );
}


void Item::setMemBuf( MemBuf *b )
{
   type( FLC_ITEM_MEMBUF );
   all.ctx.data.ptr.voidp = b;
   assignToVm( b );
}

void Item::setReference( GarbageItem *ref )
{
   type( FLC_ITEM_REFERENCE );
   all.ctx.data.ptr.voidp = ref;
   assignToVm( ref );
}

void Item::setFunction( CoreFunc* cf )
{
   type( FLC_ITEM_FUNC );
   all.ctx.data.ptr.extra = cf;
   assignToVm( cf );
}

void Item::setLBind( String *lbind, GarbageItem *val )
{
   type( FLC_ITEM_LBIND );
   all.ctx.data.ptr.voidp = lbind;
   all.ctx.data.ptr.extra = val;

   if ( lbind->isCore() )
      assignToVm( &static_cast<CoreString*>(lbind)->garbage() );

   if ( val != 0 )
       assignToVm( val );
}

void Item::setMethod( const Item &data, CoreFunc *func )
{
   *this = data;
   all.ctx.base.bits.oldType = all.ctx.base.bits.type;
   all.ctx.method = func;
   type( FLC_ITEM_METHOD );
   assignToVm( func );
}

void Item::setClassMethod( CoreObject *obj, CoreClass *cls )
{
   type( FLC_ITEM_CLSMETHOD );
   all.ctx.data.ptr.voidp = obj;
   all.ctx.data.ptr.extra = cls;
   assignToVm( obj );
   assignToVm( cls );
}

void Item::setClass( CoreClass *cls )
{
   type( FLC_ITEM_CLASS );
   // warning: class in extra to be omologue to methodClass()
   all.ctx.data.ptr.extra = cls;
   assignToVm( cls );
}

void Item::setGCPointer( FalconData *ptr, uint32 sig )
{
   type( FLC_ITEM_GCPTR );
   all.ctx.data.gptr.signature = sig;
   all.ctx.data.gptr.gcptr = new GarbagePointer( ptr );
   assignToVm( all.ctx.data.gptr.gcptr );
   ptr->gcMark( memPool->generation() );
}

void Item::setGCPointer( GarbagePointer *shell, uint32 sig )
{
   type( FLC_ITEM_GCPTR );
   all.ctx.data.gptr.signature = sig;
   all.ctx.data.gptr.gcptr = shell;
   assignToVm( shell );
   shell->ptr()->gcMark( memPool->generation() );
}

FalconData *Item::asGCPointer() const
{
   return all.ctx.data.gptr.gcptr->ptr();
}


//===========================================================================
// Generic item manipulators

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
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_LBIND:
         // methods are always filled, so they are always true.
         return true;
   }

   return false;
}

/*
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
*/

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


int64 Item::forceIntegerEx() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return asInteger();

      case FLC_ITEM_NUM:
         return (int64) asNumeric();

   }
   throw new TypeError( ErrorParam( e_param_type, __LINE__ ) );

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


bool Item::isOfClass( const String &className ) const
{
   switch( type() )
   {
      case FLC_ITEM_OBJECT:
         // objects may be classless or derived from exactly one class.
         return asObjectSafe()->derivedFrom( className );

      case FLC_ITEM_CLASS:
         return className == asClass()->symbol()->name() || asClass()->derivedFrom( className );
   }

   return false;
}


void Item::toString( String &target ) const
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

      case FLC_ITEM_RANGE:
         target = "[";
         target.writeNumber( (int64) this->asRangeStart() );
         target += ":";
         if ( ! this->asRangeIsOpen() )
         {
            target.writeNumber( (int64) this->asRangeEnd() );
            if ( ((int64) this->asRangeStart() <= (int64) this->asRangeEnd() && this->asRangeStep() != 1 ) ||
                 ((int64) this->asRangeStart() > (int64) this->asRangeEnd() && this->asRangeStep() != -1 ) )
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

      case FLC_ITEM_STRING:
         target = *asString();
      break;

      case FLC_ITEM_LBIND:
         target = "&" + *asLBind();
      break;

      case FLC_ITEM_REFERENCE:
         dereference()->toString( target );
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
         target = "Function " + this->asFunction()->symbol()->name();
      break;

      case FLC_ITEM_CLASS:
         target = "Class " + this->asClass()->symbol()->name();
      break;

      case FLC_ITEM_METHOD:
         {
            Item orig;
            this->getMethodItem( orig );
            String temp;
            orig.dereference()->toString( temp );
            target = "Method (" + temp + ")." + this->asMethodFunc()->name();
         }
      break;

      case FLC_ITEM_CLSMETHOD:
         target = "ClsMethod " + this->asMethodClass()->symbol()->name();
      break;

      default:
         target = "<?>";
   }
}

bool Item::methodize( const Item &self )
{
   Item *data = dereference();

   switch( data->type() )
   {
      case FLC_ITEM_FUNC:
      {
         data->setMethod( self, data->asFunction() );
      }
      return true;

      case FLC_ITEM_ARRAY:
         if ( data->asArray()->length() > 0 )
         {
            Item *citem = &data->asArray()->at(0);
            if ( citem->isMethod() && citem->asMethodItem() == self )
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

bool Item::isCallable() const
{
   if ( isClass() || isFunction() || isMethod() )
      return true;

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

const Item &Item::asFutureBind() const {
   return ((GarbageItem*)all.ctx.data.ptr.extra)->origin();
}

Item &Item::asFutureBind() {
   return ((GarbageItem*)all.ctx.data.ptr.extra)->origin();
}

CoreObject *Item::asObject() const {
   if ( ! isObject() )
      throw new CodeError( ErrorParam( e_static_call, __LINE__ ) );

   return (CoreObject *) all.ctx.data.ptr.voidp;
}

}

/* end of item.cpp */
