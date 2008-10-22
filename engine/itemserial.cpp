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
#include <falcon/stream.h>
#include <falcon/lineardict.h>
#include <falcon/attribute.h>
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


bool Item::serialize_function( Stream *file, const Symbol *func ) const
{
   byte type = FLC_ITEM_FUNC;
   file->write( &type, 1 );

   // we don't serialize the module ID because it is better to rebuild it on deserialization
   serialize_symbol( file, func );

   if( func->isFunction() )
   {
      FuncDef *fdef = func->getFuncDef();
      // write the called status
      uint32 itemId = fdef->onceItemId();
      if ( itemId != FuncDef::NO_STATE )
      {
         byte called = asModule()->globals().itemAt( itemId ).isNil() ? 0 : 1;
         file->write( &called, 1 );
      }

      if ( ! file->good() )
         return false;
   }

   return true;
}

bool Item::serialize_object( Stream *file, CoreObject *obj, bool bLive ) const
{
   // if we're not live, we can't have user data
   if ( ! bLive && obj->getUserData() != 0 )
   {
      obj->origin()->raiseError( e_unserializable, "Item::serialize" );
      return false;
   }

   byte type = FLC_ITEM_OBJECT;
   if ( bLive )
      type |= 0x80;

   file->write( &type, 1 );

   // write the class symbol so that it can be rebuilt back.
   serialize_symbol( file, obj->instanceOf() );

   // then serialzie the properties
   uint32 len = endianInt32( obj->propCount() );
   file->write((byte *) &len, sizeof(len) );

   for( uint32 i = 0; i < len; i ++ ) {
      Item temp;
      obj->getPropertyAt( i, temp );
      if ( temp.serialize( file, bLive ) != sc_ok )
         return false;
   }

   // and the attributes
   AttribHandler *head = obj->attributes();
   uint32 att_count = 0;
   while( head != 0 )
   {
      att_count++;
      head = head->next();
   }

   // write the size...
   len = endianInt32( att_count );
   file->write((byte *) &len, sizeof(len) );

   // and then serialize...
   head = obj->attributes();
   while( head != 0 )
   {
      head->attrib()->name().serialize( file );
      if ( ! file->good() )
         return false;

      head = head->next();
   }

   // finally, clone if live
   if ( bLive )
   {
      if ( obj->getUserData() != 0 )
      {
         void *cloned = obj->getObjectManager()->onClone( obj->origin(), obj->getUserData() );
         if ( cloned == 0 )
         {
            obj->origin()->raiseError( e_uncloneable, "Item::serialize" );
            return false;
         }
         file->write( (byte *) &cloned, sizeof( cloned ) );
      }
      else {
         void *data = 0;
         file->write( (byte *) &data, sizeof( data ) );
      }

      if ( ! file->good() )
      {
         return false;
      }
   }

   return true;
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

         int32 val1 = endianInt32(this->asRangeStart());
         int32 val2 = endianInt32(this->asRangeEnd());
         int32 val3 = endianInt32(this->asRangeStep());
         byte isOpen = this->asRangeIsOpen() ? 1 : 0;

         file->write( (byte *) &val1, sizeof( val1 ) );
         file->write( (byte *) &val2, sizeof( val2 ) );
         file->write( (byte *) &val3, sizeof( val3 ) );
         file->write( (byte *) &isOpen, sizeof( isOpen ) );
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

      case FLC_ITEM_ATTRIBUTE:
      {
         byte type = FLC_ITEM_ATTRIBUTE;
         file->write( &type, 1 );

         this->asAttribute()->name().serialize( file );
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
            type |= 0x80;

         file->write( &type, 1 );
         this->asMemBuf()->serialize( file, bLive );
      }
      break;

      case FLC_ITEM_FBOM:
      {
         byte type = FLC_ITEM_FBOM;
         file->write((byte *) &type, 1 );

         type = this->getFbomMethod();
         file->write((byte *) &type, 1 );

         Item ifbom;
         getFbomItem( ifbom );
         ifbom.serialize( file, bLive );
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

         Item key, value;
         dict->traverseBegin();
         while( dict->traverseNext( key, value ) )
         {
            key.serialize( file, bLive );
            if( ! file->good() )
               return sc_ferror;
            value.serialize( file, bLive );
            if( ! file->good() )
               return sc_ferror;
         }
      }
      break;

      case FLC_ITEM_FUNC:
         serialize_function( file, this->asFunction() );
      break;

      case FLC_ITEM_METHOD:
      {
         byte type = FLC_ITEM_METHOD;
         if ( bLive )
         {
            type |= 0x80;
            file->write( &type, 1 );
            void *obj = this->asMethodObject();
            file->write( &obj, sizeof( obj ) );
         }
         else {
            // TODO
            serialize_object( file, this->asMethodObject(), bLive );
         }
         serialize_function( file, this->asMethodFunction() );
      }
      break;

      case FLC_ITEM_TABMETHOD:
      {
         byte type = FLC_ITEM_TABMETHOD;
         file->write( &type, 1 );
         serialize_function( file, this->asMethodFunction() );

         if( isTabMethodDict() )
         {
            Item temp = asTabMethodDict();
            temp.serialize( file, bLive );
         }
         else {
            Item temp = asTabMethodArray();
            temp.serialize( file, bLive );
         }
      }
      break;

      case FLC_ITEM_OBJECT:
         serialize_object( file, this->asObject(), bLive );
      break;

      case FLC_ITEM_REFERENCE:
         asReference()->origin().serialize( file, bLive );
      break;

      case FLC_ITEM_CLASS:
         serialize_class( file, this->asClass() );
      break;

       case FLC_ITEM_CLSMETHOD:
         serialize_class( file, this->asMethodClass() );
         serialize_function( file, this->asMethodFunction() );
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
      setFunction( sym, lmod );
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
         lmod->globals().itemAt( itemId ).setInteger( 1 );
      else
         lmod->globals().itemAt( itemId ).setNil();
   }

   setFunction( sym, lmod );
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
   byte type;
   file->read((byte *) &type, 1 );

   if( ! file->good() )
      return sc_ferror;

   switch( type )
   {
      case FLC_ITEM_NIL:
         setNil();
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
         int32 val1;
         int32 val2;
         int32 val3;
         byte isOpen;

         file->read( (byte *) &val1, sizeof( val1 ) );
         file->read( (byte *) &val2, sizeof( val2 ) );
         file->read( (byte *) &val3, sizeof( val3 ) );
         file->read( (byte *) &isOpen, sizeof( isOpen ) );
         if ( file->good() ) {
            setRange( val1, val2, val3, isOpen != 0 );
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

      case FLC_ITEM_ATTRIBUTE:
      {
         String aname;

         if ( ! aname.deserialize( file ) )
         {
            return file->bad() ? sc_ferror : sc_invformat;
         }

         Attribute *attrib = vm->findAttribute( aname );
         if ( attrib == 0 )
            return sc_missclass;

         setAttribute( attrib );
      }
      break;

      case FLC_ITEM_LBIND:
      {
         int32 id;
         file->read( (byte*) &id, sizeof(id) );
         String name;
         if ( ! name.deserialize( file ) )
            return file->bad() ? sc_ferror : sc_invformat;

         setLBind( new GarbageString( vm, name ) );
      }
      break;

      case FLC_ITEM_STRING:
      {
         GarbageString *cs = new GarbageString( vm );
         setString( cs );

         if ( ! cs->deserialize( file ) )
         {
            delete cs;
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
         MemBuf *(*deserializer)( VMachine *, Stream * );
         file->read( &deserializer, sizeof( deserializer ) );
         if ( ! file->good() ) {
            return sc_ferror;
         }

         MemBuf *mb = deserializer( vm, file );
         if( mb == 0 )
         {
            return sc_invformat;
         }

         setMemBuf( mb );
         return sc_ok;
      }

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = MemBuf::deserialize( vm, file );
         if ( file->good() && mb != 0 ) {
            setMemBuf( mb );
            return sc_ok;
         }

         delete mb; // may be 0, but it's ok
         return sc_ferror;
      }
      break;


      case FLC_ITEM_FBOM:
      {
         byte methodId;
         //TODO: Verify if methodId is in a valid range.
         file->read( &methodId, 1 );

         Item bommed;
         e_sercode code = bommed.deserialize( file, vm );
         if ( code != sc_ok )
            return code;

         setFbom( bommed, methodId );
         return sc_ok;
      }

      case FLC_ITEM_ARRAY:
      {
         if( vm == 0 )
            return sc_missvm;

         int32 val;
         file->read( (byte *) &val, sizeof( val ) );
         e_sercode retval = sc_ok;

         if ( file->good() )
         {
            val = endianInt32(val);
            CoreArray *array = new CoreArray( vm, val );

            for( int i = 0; i < val; i ++ )
            {
               retval = array->elements()[i].deserialize( file, vm );
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
         if( vm == 0 )
            return sc_missvm;

         byte blessed;
         file->read( &blessed, 1 );

         int32 val;
         file->read( (byte *) &val, sizeof( val ) );

         if ( file->good() )
         {
            val = endianInt32(val);
            LinearDict *dict = new LinearDict( vm, val );
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
               dict->bless( blessed ? true : false );
               setDict( dict );

               return sc_ok;
            }

            delete  dict;
            return retval;
         }
      }
      break;


      case FLC_ITEM_FUNC:
         if( vm == 0 )
            return sc_missvm;

         return deserialize_function( file, vm );
      break;

      case FLC_ITEM_METHOD | 0x80:
      case FLC_ITEM_METHOD:
      {
         if( vm == 0 )
            return sc_missvm;

         Item obj;
         Item func;
         e_sercode sc;
         if( type == FLC_ITEM_METHOD )
         {
            // TODO
            sc = obj.deserialize( file, vm );
            if ( sc != sc_ok )
               return sc;
            if ( ! obj.isObject() )
               return sc_invformat;
         }
         else {
            CoreObject *memObj;
            file->read( &memObj, sizeof( memObj ) );
            obj = memObj;
         }

         sc = func.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;
         if ( ! func.isFunction() )
            return sc_invformat;

         setMethod( obj.asObject(), func.asFunction(), func.asModule() );
         return sc_ok;
      }

      case FLC_ITEM_TABMETHOD:
      {
         if( vm == 0 )
            return sc_missvm;

         Item vector;
         Item func;
         e_sercode sc;

         sc = func.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;
         if ( ! func.isFunction() )
            return sc_invformat;

         sc = vector.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;
         if ( vector.isArray() )
            setTabMethod( vector.asArray(), func.asFunction(), func.asModule() );
         else if ( vector.isDict() )
            setTabMethod( vector.asArray(), func.asFunction(), func.asModule() );
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

         Item *clitem = lmod->globals().itemPtrAt( sym->itemId() );

         // Create the core object, but don't fill attribs.
         CoreObject *object = clitem->asClass()->createInstance(0, false);

         // Read the class property table.
         uint32 len;
         file->read( (byte *) &len, sizeof( len ) );
         len = endianInt32( len );
         if ( len != object->propCount() ){
            return sc_missclass;
         }

         // sc must be sc_ok
         for( uint32 i = 0; i < len; i ++ ) {
            if( ! file->good() ) {
               sc = sc_ferror;
               break;
            }

            Item temp;
            sc = temp.deserialize( file, vm );
            object->setPropertyAt( i, temp );
            if ( sc != sc_ok ) {
               return sc;
            }
         }

         // now deserialize the attributes
         file->read( (byte *) &len, sizeof( len ) );
         len = endianInt32( len );
         for( uint32 j = 0; j < len; j ++ )
         {
            // first the string, which is the name...
            String a_name;
            if( ! a_name.deserialize( file ) ) {
               return file->good() ? sc_invformat : sc_ferror;
            }

            // then the attribute
            Attribute *attrib = vm->findAttribute( a_name );
            if( attrib == 0 )
            {
               return sc_missclass;
            }

            attrib->giveTo( object );
         }

         if ( sc == sc_ok ) {
            // if the object was serialized live, get the user data.
            if( bLive )
            {
               void *data;
               if ( file->read( (byte *) &data, sizeof( data ) ) != sizeof( data ) )
                  return sc_ferror;
               object->setUserData( data );
            }

            setObject( object );
            return sc_ok;
         }

         return sc;
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
bool Item::clone( Item &target, VMachine *vm ) const
{
   const Item *item = this->dereference();

   switch( item->type() )
   {
      case FLC_ITEM_STRING:
         target = new GarbageString( vm, *item->asString() );
      break;

      case FLC_ITEM_ARRAY:
         target = item->asArray()->clone();
      break;

      case FLC_ITEM_DICT:
         target = item->asDict()->clone();
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *obj = item->asObject()->clone();
         if ( obj == 0 ) {
            return false;
         }
         target = obj;
      }
      break;

      case FLC_ITEM_METHOD:
      {
         CoreObject *clone = item->asMethodObject()->clone();
         if ( clone == 0 ) {
            return false;
         }
         target.setMethod( clone, item->asMethodFunction(), item->asModule() );
      }
      break;

      case FLC_ITEM_TABMETHOD:
      {
         if( item->isTabMethodDict() )
         {
            CoreDict *clone = item->asTabMethodDict()->clone();
            if ( clone == 0 ) return false;
            target.setTabMethod( clone, item->asMethodFunction(), item->asModule() );
         }
         else {
            CoreArray *clone = item->asTabMethodArray()->clone();
            if ( clone == 0 ) return false;
            target.setTabMethod( clone, item->asMethodFunction(), item->asModule() );
         }
      }
      break;


      default:
         target = *this;
   }

   return true;
}

}


/* end of itemserial.cpp */
