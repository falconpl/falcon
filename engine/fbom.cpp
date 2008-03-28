/*
   FALCON - The Falcon Programming Language
   FILE: fbom.cpp
   $Id: fbom.cpp,v 1.10 2007/08/11 00:11:54 jonnymind Exp $

   Falcon basic object model
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer lug 4 2007
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
   Falcon basic object model
*/

#include <falcon/fbom.h>
#include <falcon/types.h>
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/vm.h>
#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/attribute.h>
#include <falcon/membuf.h>

#include <falcon/format.h>

#include <falcon/autocstring.h>
#include <falcon/bommap.h>

#include <stdio.h>

namespace Falcon {

/* BOMID: 0 */
FALCON_FUNC BOM_toString( VMachine *vm )
{
   Fbom::toString( vm, &vm->self(), vm->bomParam( 0 ) );
}

/* BOMID: 1 */
FALCON_FUNC BOM_len( VMachine *vm )
{
   Item *elem = &vm->self();
   switch( elem->type() ) {
      case FLC_ITEM_STRING:
         vm->retval( (int64) elem->asString()->length() );
      break;

      case FLC_ITEM_MEMBUF:
         vm->retval( (int64) elem->asMemBuf()->length() );
      break;

      case FLC_ITEM_ARRAY:
         vm->retval( (int64) elem->asArray()->length() );
      break;

      case FLC_ITEM_DICT:
         vm->retval( (int64) elem->asDict()->length() );
      break;

      case FLC_ITEM_ATTRIBUTE:
         vm->retval( (int64) elem->asAttribute()->size() );
      break;

      case FLC_ITEM_RANGE:
         vm->retval( 3 );
      break;

      default:
         vm->retval( 0 );
   }
}

/* BOMID: 2 */
FALCON_FUNC BOM_first( VMachine *vm )
{
   const Item &self = vm->self();
   switch( self.type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_ARRAY:
      case FLC_ITEM_DICT:
      case FLC_ITEM_ATTRIBUTE:
         Fbom::makeIterator( vm, self, true );
      break;

      case FLC_ITEM_RANGE:
         vm->retval( (int64) self.asRangeStart() );
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_prop_acc ) ) );
   }
}

/* BOMID: 3 */
FALCON_FUNC BOM_last( VMachine *vm )
{
   const Item &self = vm->self();

   switch( self.type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_ARRAY:
      case FLC_ITEM_DICT:
      // attributes cannot be scanned backwards
         Fbom::makeIterator( vm, self, false );
      break;

      case FLC_ITEM_RANGE:
         if( self.asRangeIsOpen() )
            vm->retnil();
         else
            vm->retval( (int64) self.asRangeEnd() );

      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_prop_acc ) ) );
   }
}

/* BOMID: 4 */
FALCON_FUNC BOM_compare( VMachine *vm )
{
   if( vm->bomParamCount() == 0 )
   {
       vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
       return;
   }

   Item *comparand = vm->bomParam( 0 );
   vm->retval( vm->self().compare( *comparand ) );

}

/* BOMID: 5 */
FALCON_FUNC BOM_equal( VMachine *vm )
{
   if( vm->bomParamCount() == 0 )
   {
       vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
       return;
   }

   Item *comparand = vm->bomParam( 0 );
   vm->retval( vm->self().equal( *comparand ) );

}

/* BOMID: 6 */
FALCON_FUNC BOM_type( VMachine *vm )
{
   vm->retval( vm->self().type() );
}

/* BOMID: 7 */
FALCON_FUNC BOM_className( VMachine *vm )
{
   if( vm->self().isObject() )
   {
      vm->retval(
         new GarbageString( vm, vm->self().asObject()->instanceOf()->name() ) );
   }
   else if( vm->self().isClass() )
   {
      vm->retval(
         new GarbageString( vm, vm->self().asClass()->symbol()->name() ) );
   }
   else {
      vm->retnil();
   }
}

/* BOMID: 8 */
FALCON_FUNC BOM_baseClass( VMachine *vm )
{

   if( vm->self().isObject() )
   {
      Symbol *cls = vm->self().asObject()->instanceOf();
      Item *i_cls = vm->findLocalSymbolItem( cls->name() );

      if( i_cls != 0 )
         vm->retval( *i_cls );
      else
         vm->retnil();
   }
   else {
      vm->retnil();
   }
}

/* BOMID: 9 */
FALCON_FUNC BOM_derivedFrom( VMachine *vm )
{
   Item *clsName = vm->bomParam( 0 );

   if( clsName == 0 || ! clsName->isString() )
   {
       vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "S" ) ) );
       return;
   }

   String *name = clsName->asString();
   if( vm->self().isObject() )
   {
      vm->retval( vm->self().asObject()->derivedFrom( *name ) );
   }
   else if( vm->self().isClass() )
   {
      vm->retval( vm->self().asClass()->derivedFrom( *name ) );
   }
   else {
      vm->retval( (int64) 0 );
   }
}

/* BOMID: 10 */
FALCON_FUNC BOM_clone( VMachine *vm )
{
   if ( ! vm->self().clone( vm->regA(), vm ) )
   {
      vm->raiseError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
}

/* BOMID: 11 */
FALCON_FUNC BOM_serialize( VMachine *vm )
{
   Item *fileId = vm->bomParam( 0 );
   Item *source = vm->self().dereference();

   if( fileId == 0 || ! fileId->isObject() || ! fileId->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "O:Stream" ) ) );
      return;
   }

   Stream *file = (Stream *) fileId->asObject()->getUserData();
   Item::e_sercode sc = source->serialize( file, vm );
   switch( sc )
   {
      case Item::sc_ok: vm->retval( 1 ); break;
      case Item::sc_ferror:
         vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_vm ) ) );
      default:
         vm->retnil(); // VM may already have raised an error.
         //TODO: repeat error.
   }

}

/* BOMID: 12 */
FALCON_FUNC BOM_attribs( VMachine *vm )
{
   Item *source = vm->self().dereference();
   if( source->isObject() )
   {
      CoreArray *array = new CoreArray( vm );
      AttribHandler *attribs = source->asObject()->attributes();
      while( attribs != 0 )
      {
         array->append( attribs->attrib() );
         attribs = attribs->next();
      }
      vm->retval( array );
   }
   else
      vm->retnil();
}

/* BOMID: 13 */
FALCON_FUNC BOM_backTrim( VMachine *vm )
{
   const Item &self = vm->self();
   if ( self.type() == FLC_ITEM_STRING ) {
      String s = *self.asString();
      s.backTrim();
      vm->retval( s );
   } else {
      vm->raiseRTError( new ParamError( ErrorParam( e_prop_acc ) ) );
   }
}

/* BOMID: 14 */
FALCON_FUNC BOM_frontTrim( VMachine *vm )
{
   const Item &self = vm->self();
   if ( self.type() == FLC_ITEM_STRING ) {
      String s = *self.asString();
      s.frontTrim();
      vm->retval( s );
   } else {
      vm->raiseRTError( new ParamError( ErrorParam( e_prop_acc ) ) );
   }
}

/* BOMID: 15 */
FALCON_FUNC BOM_allTrim( VMachine *vm )
{
   const Item &self = vm->self();
   if ( self.type() == FLC_ITEM_STRING ) {
      String s = *self.asString();
      s.trim();
      vm->retval( s );
   } else {
      vm->raiseRTError( new ParamError( ErrorParam( e_prop_acc ) ) );
   }
}

//====================================================//
// THE BOM TABLE
//====================================================//


static void (* const  BOMTable  [] ) ( Falcon::VMachine *) =
{
   BOM_toString,
   BOM_len,
   BOM_first,
   BOM_last,
   BOM_compare,
   BOM_equal,
   BOM_type,
   BOM_className,
   BOM_baseClass,
   BOM_derivedFrom,
   BOM_clone,
   BOM_serialize,
   BOM_attribs,
   BOM_backTrim,
   BOM_frontTrim,
   BOM_allTrim
};

//====================================================//
// THE BOM IMPLEMENTATION
//====================================================//

bool Item::getBom( const String &property, Item &method, BomMap *bmap ) const
{
   int *value = (int *) bmap->find( &property );
   if ( value == NULL )
      return false;
   method.setFbom( *this, *value );
   return true;
}


bool Item::callBom( VMachine *vm ) const
{
   if( isFbom() )
   {
      // switching here for type may allow to create different BOMs tables for each item type.

      // Switching self/sender. Is it needed?
      Item oldSender = vm->sender();
      vm->sender() = vm->self();

      // real call
      getFbomItem( vm->self() );
      // todo: check validity
      void (* const f)( VMachine *)  = BOMTable[ getFbomMethod() ];
      f( vm );

      // Switching self/sender. Is it needed?
      vm->self() = vm->sender();
      vm->sender() = oldSender;

      return true;
   }

   return false;
}

//====================================================//
// THE BOM UTILITIES
//====================================================//

namespace Fbom {

void toString( VMachine *vm, Item *elem, Item *format )
{

   if ( elem != 0 )
   {

      GarbageString *ret = new GarbageString( vm );

      if ( format != 0 )
      {
         if ( format->isString() )
         {
            Format fmt( *format->asString() );
            if( ! fmt.isValid() )
            {
               vm->raiseRTError( new ParamError( ErrorParam( e_param_fmt_code ) ) );
               return;
            }

            if ( fmt.format( vm, *elem, *ret ) )
            {
               vm->retval( ret );
               return;
            }
         }
         else if ( format->isObject() )
         {
            CoreObject *fmtO = format->asObject();
            if( fmtO->derivedFrom( "Format" ) )
            {
               Format *fmt = static_cast<Format *>( fmtO->getUserData() );
               if ( fmt->format( vm, *elem, *ret ) )
               {
                  vm->retval( ret );
                  return;
               }
            }
         }
      }
      else {
         vm->itemToString( *ret, elem, "" );
         vm->retval( ret );
         return;
      }
   }

   vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X,[S/O]" ) ) );
}


void makeIterator( VMachine *vm, const Item &self, bool begin )
{
   // create the iterator
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   switch( self.type() )
   {
      case FLC_ITEM_STRING:
      {
         String *orig = self.asString();
         int64 pos = begin ? 0 : (orig->size() == 0 ? 0 : orig->length() - 1);
         iterator->setProperty( "_pos", pos );
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *orig = self.asMemBuf();
         int64 pos = begin ? 0 : (orig->size() == 0 ? 0 : orig->length() - 1);
         iterator->setProperty( "_pos", pos );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *orig = self.asArray();
         int64 pos = begin ? 0 : (orig->length() == 0 ? 0 : orig->length() - 1);
         iterator->setProperty( "_pos", pos );
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *orig = self.asDict();
         DictIterator *iter;
         if( begin )
            iter = orig->first();
         else
            iter = orig->last();
         iterator->setUserData( iter );
      }
      break;

      case FLC_ITEM_ATTRIBUTE:
      {
         Attribute *attrib = self.asAttribute();
         // only from begin.
         iterator->setUserData( attrib->getIterator() );
      }
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ) ) );
         return;
   }

   iterator->setProperty( "_origin", self );
   vm->retval( iterator );
}


} // fbom
} // falcon

/* end of fbom.cpp */
