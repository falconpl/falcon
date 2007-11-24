/*
   FALCON - The Falcon Programming Language
   FILE: itemserial.cpp
   $Id: itemserial.cpp,v 1.12 2007/08/11 00:11:54 jonnymind Exp $

   Item serialization methods
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom gen 28 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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


bool Item::serialize_function( Stream *file, const Symbol *func, VMachine *vm ) const
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
         byte called = vm->moduleItem( asModuleId(), itemId ).isNil() ? 0 : 1;
         file->write( &called, 1 );
      }

      if ( ! file->good() )
         return false;
   }

   return true;
}

bool Item::serialize_object( Stream *file, const CoreObject *obj, VMachine *vm ) const
{
   byte type = FLC_ITEM_OBJECT;
   file->write( &type, 1 );

   // write the class symbol so that it can be rebuilt back.
   serialize_symbol( file, obj->instanceOf() );

   // then serialzie the properties
   uint32 len = endianInt32( obj->propCount() );
   file->write((byte *) &len, sizeof(len) );

   for( uint32 i = 0; i < len; i ++ ) {
      if ( obj->getPropertyAt( i ).serialize( file, vm ) != sc_ok )
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

   return true;
}


Item::e_sercode Item::serialize( Stream *file, VMachine *vm ) const
{
   if( file->bad() )
      return sc_ferror;

   switch( this->type() )
   {
      case FLC_ITEM_INT:
      {
         char type = FLC_ITEM_INT;
         file->write((byte *) &type, 1 );

         int64 val = endianInt64( this->asInteger() );
         file->write( (byte *) &val, sizeof( val ) );
      }
      break;

      case FLC_ITEM_RANGE:
      {
         char type = FLC_ITEM_RANGE;
         file->write((byte *) &type, 1 );

         int32 val1 = endianInt32(this->asRangeStart());
         int32 val2 = endianInt32(this->asRangeEnd());
         byte isOpen = this->asRangeIsOpen() ? 1 : 0;

         file->write( (byte *) &val1, sizeof( val1 ) );
         file->write( (byte *) &val2, sizeof( val2 ) );
         file->write( (byte *) &isOpen, sizeof( isOpen ) );
      }
      break;

      case FLC_ITEM_NUM:
      {
         char type = FLC_ITEM_NUM;
         file->write((byte *) &type, 1 );

         numeric val = endianNum( this->asNumeric() );
         file->write( (byte *) &val, sizeof( val ) );
      }
      break;

      case FLC_ITEM_ATTRIBUTE:
      {
         char type = FLC_ITEM_ATTRIBUTE;
         file->write((byte *) &type, 1 );

         this->asAttribute()->name().serialize( file );
      }
      break;

      case FLC_ITEM_STRING:
      {
         char type = FLC_ITEM_STRING;
         file->write((byte *) &type, 1 );

         this->asString()->serialize( file );
      }
      break;

      case FLC_ITEM_FBOM:
      {
         char type = FLC_ITEM_FBOM;
         file->write((byte *) &type, 1 );

         type = this->getFbomMethod();
         file->write((byte *) &type, 1 );

         Item ifbom;
         getFbomItem( ifbom );
         ifbom.serialize( file, vm );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         char type = FLC_ITEM_ARRAY;
         file->write((byte *) &type, 1 );

         CoreArray *array = this->asArray();
         int32 len = endianInt32( array->length() );
         file->write( (byte *) &len, sizeof( len ) );
         for( uint32 i = 0; i < array->length(); i ++ ) {
            array->elements()[i].serialize( file, vm );
            if( ! file->good() )
               return sc_ferror;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         char type = FLC_ITEM_DICT;
         file->write((byte *) &type, 1 );

         CoreDict *dict = this->asDict();
         int32 len = endianInt32( dict->length() );
         file->write( (byte *) &len, sizeof( len ) );

         Item key, value;
         dict->traverseBegin();
         while( dict->traverseNext( key, value ) )
         {
            key.serialize( file, vm );
            if( ! file->good() )
               return sc_ferror;
            value.serialize( file, vm );
            if( ! file->good() )
               return sc_ferror;
         }
      }
      break;

      case FLC_ITEM_FUNC:
         serialize_function( file, this->asFunction(), vm );
      break;

      case FLC_ITEM_METHOD:
      {
         byte type = FLC_ITEM_METHOD;
         file->write( &type, 1 );

         serialize_object( file, this->asMethodObject(), vm );
         serialize_function( file, this->asMethodFunction(), vm );
      }
      break;

      case FLC_ITEM_OBJECT:
         serialize_object( file, this->asObject(), vm );
      break;

      case FLC_ITEM_REFERENCE:
         asReference()->origin().serialize( file, vm );
      break;

      default:
      {
         char type = FLC_ITEM_NIL;
         file->write((byte *) &type, 1 );
      }
   }

   return file->bad() ? sc_ferror : sc_ok;
}


Item::e_sercode Item::deserialize_symbol( Stream *file, VMachine *vm, Symbol **tg_sym, int16 *modId )
{
   if ( vm == 0 )
      return sc_missvm;

   // read the module name
   String name;
   if ( ! name.deserialize( file ) )
      return sc_ferror;

   *modId = vm->getModuleId( name );
   if ( *modId == -1 ) {
      return sc_misssym;
   }

   const Module *mod = vm->getModule( *modId );

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
   int16 modId;

   e_sercode sc = deserialize_symbol( file, vm, &sym, &modId );
   if ( sc != sc_ok  )
      return sc;

   // read the function called status

   if ( ! sym->isFunction() ) {
      // external function
      setFunction( sym, modId );
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
         vm->moduleItem( modId, itemId ).setInteger( 1 );
      else
         vm->moduleItem( modId, itemId ).setNil();
   }

   setFunction( sym, modId );
   return sc_ok;
}


Item::e_sercode Item::deserialize( Stream *file, VMachine *vm )
{
   char type;
   file->read((byte *) &type, 1 );

   if( ! file->good() )
      return sc_ferror;

   switch( type )
   {
      case FLC_ITEM_NIL:
         setNil();
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
         byte isOpen;

         file->read( (byte *) &val1, sizeof( val1 ) );
         file->read( (byte *) &val2, sizeof( val2 ) );
         file->read( (byte *) &isOpen, sizeof( isOpen ) );
         if ( file->good() ) {
            setRange( val1, val2, isOpen != 0 );
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

      case FLC_ITEM_METHOD:
      {
         if( vm == 0 )
            return sc_missvm;

         Item obj, func;
         e_sercode sc = obj.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;
         if ( ! obj.isObject() )
            return sc_invformat;

         sc = func.deserialize( file, vm );
         if ( sc != sc_ok )
            return sc;
         if ( ! func.isFunction() )
            return sc_invformat;

         setMethod( obj.asObject(), func.asFunction(), func.asModuleId() );
         return sc_ok;
      }

      case FLC_ITEM_OBJECT:
      {
         if( vm == 0 )
            return sc_missvm;

         // read the module name
         Symbol *sym;
         int16 modId;
         e_sercode sc = deserialize_symbol( file, vm, &sym, &modId );
         if ( sc != sc_ok  )
            return sc;

         Item *clitem = &vm->moduleItem( modId, sym->itemId() );

         // Create the core object, but don't fill attribs.
         CoreObject *object = clitem->asClass()->createInstance(false);

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

            sc = object->getPropertyAt( i ).deserialize( file, vm );
            if ( sc != sc_ok ) {
               break;
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
            setObject( object );
            return sc_ok;
         }

         return sc;
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
         target.setMethod( clone, item->asMethodFunction(), item->asModuleId() );
      }
      break;


      default:
         target = *this;
   }

   return true;
}

}


/* end of itemserial.cpp */
