/*
   FALCON - The Falcon Programming Language
   FILE: itemserial.cpp

   Item serialization methods
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom gen 28 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Item serialization methods.
   The serialization methods have been put in this file as they require
   also linking with the stream class.
*/

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/corefunc.h>
#include <falcon/stream.h>
#include <falcon/lineardict.h>
#include <falcon/membuf.h>

namespace Falcon {

bool Item::serialize_symbol( Stream *file, const Symbol *sym ) const
{
   const Module *mod= sym->module();

   // write the module name
   mod->name().serialize( file );

   // write the symbol name
   sym->name().serialize( file );

   if( ! file->good() )
      return false;
   return true;
}


bool Item::serialize_function( Stream *file, const CoreFunc *func, bool bLive ) const
{
   byte type = FLC_ITEM_FUNC;
   if ( bLive )
         type |= 0x80;
   file->write( &type, 1 );

   // we don't serialize the module ID because it is better to rebuild it on deserialization
   serialize_symbol( file, func->symbol() );

   if ( func->symbol()->isFunction() )
   {
      // language function ? -- serialize the state.
      FuncDef *fdef = func->symbol()->getFuncDef();
      // write the called status
      uint32 itemId = fdef->onceItemId();
      if ( itemId != FuncDef::NO_STATE )
      {
         byte called = func->liveModule()->globals()[ itemId ].isNil() ? 0 : 1;
         file->write( &called, 1 );
      }

      // and we may need to serialize also it's closure.
      ItemArray* closed = func->closure();
      if( closed != 0 )
      {
         int32 len = endianInt32( closed->length() );
         file->write( (byte *) &len, sizeof( len ) );

         for( uint32 i = 0; i < closed->length(); i ++ )
         {
            (*closed)[i].serialize( file, bLive );
            if( ! file->good() )
               return false;
         }
      }
      else {
         int32 len = 0;
         file->write( (byte *) &len, sizeof( len ) );
      }
   }

   if ( ! file->good() )
      return false;

   return true;
}

bool Item::serialize_object( Stream *file, CoreObject *obj, bool bLive ) const
{
   byte type = FLC_ITEM_OBJECT;
   if ( bLive )
      type |= 0x80;

   file->write( &type, 1 );

   // write the class symbol so that it can be rebuilt back.
   serialize_symbol( file, obj->generator()->symbol() );

   // then serialzie the object itself
   return obj->serialize( file, bLive );
}


bool Item::serialize_class( Stream *file, const CoreClass *cls ) const
{
   byte type = FLC_ITEM_CLASS;
   file->write( &type, 1 );

   LiveModule *lmod = cls->liveModule();

   // Write the live module name
   lmod->name().serialize( file );
   if ( ! file->good() )
         return false;

   // and the class name
   cls->symbol()->name().serialize( file );
   if ( ! file->good() )
         return false;

   return true;
}


Item::e_sercode Item::serialize( Stream *file, bool bLive ) const
{
   if( file->bad() )
      return sc_ferror;

   switch( this->type() )
   {
      case FLC_ITEM_BOOL:
      {
         byte type = FLC_ITEM_BOOL;
         file->write((byte *) &type, 1 );

         byte bval = this->asBoolean() ? 1 : 0;
         file->write( (byte *) &bval, sizeof( bval ) );
      }
      break;

      case FLC_ITEM_INT:
      {
         byte type = FLC_ITEM_INT;
         file->write((byte *) &type, 1 );

         int64 val = endianInt64( this->asInteger() );
         file->write( (byte *) &val, sizeof( val ) );
      }
      break;

      case FLC_ITEM_RANGE:
      {
         byte type = FLC_ITEM_RANGE;
         file->write((byte *) &type, 1 );

         int64 val1 = endianInt64(this->asRangeStart());
         int64 val2 = endianInt64(this->asRangeEnd());
         int64 val3 = endianInt64(this->asRangeStep());
         //byte isOpen = this->asRangeIsOpen() ? 1 : 0;

         file->write( (byte *) &val1, sizeof( val1 ) );
         file->write( (byte *) &val2, sizeof( val2 ) );
         file->write( (byte *) &val3, sizeof( val3 ) );
         //file->write( (byte *) &isOpen, sizeof( isOpen ) );
      }
      break;

      case FLC_ITEM_NUM:
      {
         byte type = FLC_ITEM_NUM;
         file->write((byte *) &type, 1 );

         numeric val = endianNum( this->asNumeric() );
         file->write( (byte *) &val, sizeof( val ) );
      }
      break;

      case FLC_ITEM_STRING:
      {
         byte type = FLC_ITEM_STRING;
         file->write((byte *) &type, 1 );
         this->asString()->serialize( file );
      }
      break;

      case FLC_ITEM_LBIND:
      {
         byte type = FLC_ITEM_LBIND;
         file->write((byte *) &type, 1 );
         // Future bindings are temporary items; as such, their future
         // value is not to be serialized.
         asLBind()->serialize( file );
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         byte type = FLC_ITEM_MEMBUF;
         if ( bLive )
         {
            type |= 0x80;
            file->write( &type, 1 );
            MemBuf* mb = asMemBuf();
            file->write( &mb, sizeof(mb) );
         }
         else {
            file->write( &type, 1 );
            this->asMemBuf()->serialize( file, bLive );
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         byte type = FLC_ITEM_ARRAY;
         file->write((byte *) &type, 1 );

         CoreArray &array = *this->asArray();
         int32 len = endianInt32( array.length() );
         file->write( (byte *) &len, sizeof( len ) );
         for( uint32 i = 0; i < array.length(); i ++ ) {
            array[i].serialize( file, bLive );
            if( ! file->good() )
               return sc_ferror;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         byte type = FLC_ITEM_DICT;
         file->write( &type, 1 );

         CoreDict *dict = this->asDict();
         type = dict->isBlessed() ? 1:0;
         file->write( &type, 1 );

         int32 len = endianInt32( dict->length() );
         file->write( (byte *) &len, sizeof( len ) );

         Iterator iter( &dict->items() );
         while( iter.hasCurrent() )
         {
            iter.getCurrentKey().serialize( file, bLive );
            if( ! file->good() )
               return sc_ferror;
            iter.getCurrent().serialize( file, bLive );
            if( ! file->good() )
               return sc_ferror;

            iter.next();
         }
      }
      break;

      case FLC_ITEM_FUNC:
         serialize_function( file, this->asFunction(), bLive );
      break;

      case FLC_ITEM_METHOD:
      {
         byte type = FLC_ITEM_METHOD;
         file->write( &type, 1 );

         e_sercode sc = asMethodItem().serialize( file, bLive );
         if( sc != sc_ok )
            return sc;

         CallPoint* cp = this->asMethodFunc();
         if ( cp->isFunc() )
         {
            serialize_function( file, static_cast<CoreFunc*>(cp), bLive );
         }
         else
         {
            SafeItem arr( static_cast<CoreArray*>(cp) );
            arr.serialize(file, bLive );
         }
      }
      break;

      case FLC_ITEM_OBJECT:
         serialize_object( file, this->asObjectSafe(), bLive );
      break;

      case FLC_ITEM_REFERENCE:
         asReference()->origin().serialize( file, bLive );
      break;

      case FLC_ITEM_CLASS:
         serialize_class( file, this->asClass() );
      break;

      case FLC_ITEM_CLSMETHOD:
         serialize_class( file, this->asMethodClass() );
         serialize_function( file, this->asFunction(), bLive );
      break;

      case FLC_ITEM_UNB:
      {
         byte type = FLC_ITEM_UNB;
         file->write((byte *) &type, 1 );
      }
      break;

      default:
      {
         byte type = FLC_ITEM_NIL;
         file->write((byte *) &type, 1 );
      }
   }

   return file->bad() ? sc_ferror : sc_ok;
}


Item::e_sercode Item::deserialize_symbol( Stream *file, VMachine *vm, Symbol **tg_sym, LiveModule **livemod )
{
   if ( vm == 0 )
      return sc_missvm;

   // read the module name
   String name;
   if ( ! name.deserialize( file ) )
      return sc_ferror;

   LiveModule *lmod = vm->findModule( name );
   *livemod = lmod;
   if ( lmod == 0 ) {
      return sc_misssym;
   }

   const Module *mod = lmod->module();

   // read the symbol name
   if ( ! name.deserialize( file ) )
      return file->bad() ? sc_ferror : sc_invformat;

   // find the name in the module
   Symbol *sym = mod->findGlobalSymbol( name );
   if ( sym == 0 ) {
      return sc_misssym;
   }

   *tg_sym = sym;
   return sc_ok;
}



Item::e_sercode Item::deserialize_function( Stream *file, VMachine *vm )
{
   if ( vm == 0 )
      return sc_missvm;

   Symbol *sym;
   LiveModule *lmod;

   e_sercode sc = deserialize_symbol( file, vm, &sym, &lmod );
   if ( sc != sc_ok  )
      return sc;

   // read the function called status

   if ( ! sym->isFunction() ) {
      // external function
      setFunction( new CoreFunc( sym, lmod ) );
      return sc_ok;
   }

   // internal function.
   FuncDef *def = sym->getFuncDef();

   // read the once status
   uint32 itemId = def->onceItemId();
   if( itemId != FuncDef::NO_STATE )
   {
      byte called;
      file->read( &called, 1 );
      if( called )
         lmod->globals()[ itemId ].setInteger( 1 );
      else
         lmod->globals()[ itemId ].setNil();
   }

   CoreFunc* function = new CoreFunc( sym, lmod );
   // read the closure state.
   uint32 closlen;
   file->read( &closlen, sizeof( closlen ) );
   if( closlen != 0 )
   {
      ItemArray* closure = new ItemArray( closlen );
      closure->resize( closlen );

      // put the state in now, so we can destroy it in case of error.
      function->closure( closure );

      // deserialize all the items.
      for( uint32 i = 0; i < closlen; i++ ) {
         Item::e_sercode res = (*closure)[i].deserialize(file, vm );
         if ( res != sc_ok )
            return res;
      }

   }

   setFunction( function );
   return sc_ok;
}

Item::e_sercode Item::deserialize_class( Stream *file, VMachine *vm )
{
   if ( vm == 0 )
      return sc_missvm;

   String modName, className;
   if ( ! modName.deserialize( file ) || ! className.deserialize( file ) )
      return file->bad() ? sc_ferror : sc_invformat;

   // find the module in the vm
   LiveModule *origMod = vm->findModule( modName );
   if( origMod == 0 )
   {
      return sc_misssym;
   }

   // find the class item in the module
   Item *clitem = origMod->findModuleItem( className );
   if ( clitem == 0 )
      return sc_misssym;

   if ( clitem->isReference() )
   {
      if( ! clitem->dereference()->isClass() )
         return sc_misssym;
   }
   else if ( ! clitem->isClass() )
      return sc_misssym;

   *this = *clitem;
   return sc_ok;
}

Item::e_sercode Item::deserialize( Stream *file, VMachine *vm )
{
   byte type = FLC_ITEM_NIL;
   if ( file->read((byte *) &type, 1 ) == 0 )
      return sc_eof;

   if( ! file->good() )
      return sc_ferror;

   switch( type )
   {
      case FLC_ITEM_NIL:
         setNil();
      return sc_ok;

      case FLC_ITEM_UNB:
         setUnbound();
         return sc_ok;

      case FLC_ITEM_BOOL:
      {
         byte bval;
         file->read( (byte *) &bval, sizeof( bval ) );
         if ( file->good() ) {
            setBoolean( bval != 0 );
            return sc_ok;
         }
         return sc_ferror;
      }
      return sc_ok;

      case FLC_ITEM_INT:
      {
         int64 val;
         file->read( (byte *) &val, sizeof( val ) );
         if ( file->good() ) {
            setInteger(endianInt64(val) );
            return sc_ok;
         }
         return sc_ferror;
      }
      break;

      case FLC_ITEM_RANGE:
      {
         int64 val1;
         int64 val2;
         int64 val3;
         //byte isOpen;

         file->read( (byte *) &val1, sizeof( val1 ) );
         file->read( (byte *) &val2, sizeof( val2 ) );
         file->read( (byte *) &val3, sizeof( val3 ) );
         //file->read( (byte *) &isOpen, sizeof( isOpen ) );

         val1 = endianInt64( val1 );
         val2 = endianInt64( val2 );
         val3 = endianInt64( val3 );

         if ( file->good() ) {
            setRange( new CoreRange( val1, val2, val3 ) );
            return sc_ok;
         }
         return sc_ferror;
      }
      break;

      case FLC_ITEM_NUM:
      {
         numeric val;
         file->read( (byte *) &val, sizeof( val ) );
         if ( file->good() ) {
            setNumeric( endianNum( val ) );
            return sc_ok;
         }
         return sc_ferror;
      }
      break;

      case FLC_ITEM_LBIND:
      {
         int32 id;
         file->read( (byte*) &id, sizeof(id) );
         String name;
         if ( ! name.deserialize( file ) )
            return file->bad() ? sc_ferror : sc_invformat;

         setLBind( new CoreString( name ) );
      }
      break;

      case FLC_ITEM_STRING:
      {
         CoreString *cs = new CoreString;
         setString( cs );

         if ( ! cs->deserialize( file ) )
         {
            return file->bad() ? sc_ferror : sc_invformat;
         }

         if ( file->good() ) {
            return sc_ok;
         }

         return sc_ferror;
      }
      break;

      case FLC_ITEM_MEMBUF |0x80:
      {
         // get the function pointer in the stream
         /*MemBuf *(*deserializer)( VMachine *, Stream * );
         file->read( &deserializer, sizeof( deserializer ) );
         if ( ! file->good() ) {
            return sc_ferror;
         }

         MemBuf *mb = deserializer( vm, file );
         if( mb == 0 )
         {
            return sc_invformat;
         }*/

         MemBuf* mb;
         if( file->read( &mb, sizeof( mb ) ) == sizeof(mb) )
         {
            setMemBuf( mb );
            return sc_ok;
         }
         return sc_eof;
      }

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = MemBuf::deserialize( vm, file );
         if ( file->good() && mb != 0 ) {
            setMemBuf( mb );
            return sc_ok;
         }

         return sc_ferror;
      }
      break;


      case FLC_ITEM_ARRAY:
      {
         int32 val;
         file->read( (byte *) &val, sizeof( val ) );
         e_sercode retval = sc_ok;

         if ( file->good() )
         {
            val = endianInt32(val);
            CoreArray *array = new CoreArray( val );

            for( int i = 0; i < val; i ++ )
            {
               retval = array->items()[i].deserialize( file, vm );
               if( retval != sc_ok ) {
                  break;
               }
               array->length( i + 1 );
            }

            if ( retval == sc_ok ) {
               setArray( array );
               return sc_ok;
            }

            return retval;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         byte blessed;
         file->read( &blessed, 1 );

         int32 val;
         file->read( (byte *) &val, sizeof( val ) );

         if ( file->good() )
         {
            val = endianInt32(val);
            LinearDict *dict = new LinearDict( val );
            LinearDictEntry *elems = dict->entries();
            e_sercode retval = sc_ok;
            for( int i = 0; i < val; i ++ ) {
               LinearDictEntry *entry = elems + i;
               retval = entry->key().deserialize( file, vm );
               if( retval == sc_ok )
                    retval = entry->value().deserialize( file, vm );

               if ( retval != sc_ok )
                  break;
               dict->length( i + 1 );
            }

            if( retval == sc_ok ) {
               CoreDict* cdict = new CoreDict( dict );
               cdict->bless( blessed ? true : false );
               setDict( cdict );

               return sc_ok;
            }
            else
               delete dict;

            return retval;
         }
      }
      break;

      case FLC_ITEM_FUNC | 0x80:
      case FLC_ITEM_FUNC:
      {
         if( vm == 0 )
            return sc_missvm;

         return deserialize_function( file, vm );
      }
     break;

      case FLC_ITEM_METHOD:
      {
         if( vm == 0 )
            return sc_missvm;

         Item obj;
         Item func;
         e_sercode sc;
         sc = obj.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;

         sc = func.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;
         if ( func.isFunction() )
            setMethod( obj, func.asMethodFunc() );
         else if ( func.isArray() && func.isCallable() )
         {
            setMethod( obj, func.asArray() );
         }
         else
            return sc_invformat;

         return sc_ok;
      }


      case FLC_ITEM_OBJECT | 0x80:
      case FLC_ITEM_OBJECT:
      {
         bool bLive = type != FLC_ITEM_OBJECT;

         if( vm == 0 )
            return sc_missvm;

         // read the module name
         Symbol *sym;
         LiveModule *lmod;
         e_sercode sc = deserialize_symbol( file, vm, &sym, &lmod );
         if ( sc != sc_ok  )
            return sc;

         Item *clitem = &lmod->globals()[ sym->itemId() ];

         // Create the core object, but don't fill attribs.
         CoreObject *object = clitem->dereference()->asClass()->createInstance(0, true);
         if ( ! object->deserialize( file, bLive ) )
         {
            return sc_missclass;
         }

         setObject( object );
         return file->good() ? sc_ok : sc_ferror;
      }
      break;

      case FLC_ITEM_CLASS:
         return deserialize_class( file, vm );


       case FLC_ITEM_CLSMETHOD:
       {
         e_sercode sc = deserialize_class( file, vm );
         if ( sc != sc_ok )
            return sc;
         return deserialize_function( file, vm );
      }
      break;

      default:
         return sc_invformat;
   }

   return sc_ferror;
}


//TODO:Move in another file.
bool Item::clone( Item &target) const
{
   const Item *item = this->dereference();

   switch( item->type() )
   {
      case FLC_ITEM_STRING:
         target = new CoreString( *item->asString() );
      break;

      case FLC_ITEM_ARRAY:
         target = item->asArray()->clone();
      break;

      case FLC_ITEM_DICT:
         target = item->asDict()->clone();
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *obj = item->asObjectSafe()->clone();
         if ( obj == 0 ) {
            return false;
         }
         target = obj;
      }
      break;

      case FLC_ITEM_METHOD:
      {
         target = *this;
      }
      break;


      default:
         target = *this;
   }

   return true;
}
/*
void Item::copy( const Item &other )
{
   if ( isLBind() )
   {
      VMachine* current = VMachine::getCurrent();
      if ( current != 0 )
      {
         current->bindItem( *asLBind(), other );
         return;
      }

   }

   if( other.isLBind() )
   {
      VMachine* current = VMachine::getCurrent();
      if ( current != 0 )
      {
         current->unbindItem( *other.asLBind(), *this );
         return;
      }
   }

   all = other.all;
}
*/

}


/* end of itemserial.cpp */
