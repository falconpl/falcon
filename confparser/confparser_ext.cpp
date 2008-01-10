/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp
   $Id: confparser_ext.cpp,v 1.14 2007/08/11 00:11:55 jonnymind Exp $

   Falcon VM interface to confparser module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50
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
   Falcon VM interface to socket module.
*/

#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/lineardict.h>
#include <falcon/stream.h>
#include <falcon/memory.h>

#include "confparser_mod.h"
namespace Falcon {
namespace Ext {

// ==============================================
// Class ConfParser
// ==============================================

/**

*/
FALCON_FUNC  ConfParser_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_fname = vm->param(0);
   Item *i_encoding = vm->param(1);

   if ( (i_fname != 0 && ! i_fname->isString()) || ( i_encoding != 0 && ! i_encoding->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, [S]" ) ) );
      return;
   }

   String fname;
   String encoding;

   if ( i_fname != 0 )
      fname = *i_fname->asString();

   if ( i_encoding != 0 )
      encoding = *i_encoding->asString();

   ConfigFile *cfile = new ConfigFile( fname, encoding );
   self->setUserData( cfile );
}


FALCON_FUNC  ConfParser_read( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_stream = vm->param(0);

   bool bRes;

   if( i_stream == 0 )
   {
      bRes = cfile->load();
   }
   else {
      bool bValid = false;
      if ( i_stream->isObject() )
      {
         CoreObject *streamObj = i_stream->asObject();
         if ( streamObj->derivedFrom( "Stream" ) )
         {
            Stream *base = (Stream *) streamObj->getUserData();
            bRes = cfile->load( base );
            bValid = true;
         }
      }

      if ( ! bValid )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "Stream" ) ) );
         return;
      }
   }

   if ( ! bRes )
   {
      String msg = cfile->errorMessage() + " at ";
      msg.writeNumber( (int64) cfile->errorLine() );
      vm->raiseModError( new ParseError( ErrorParam( 1260, __LINE__ ).
         desc( "Error parsing the file" ).extra( msg ) ) );
      self->setProperty( "error", cfile->errorMessage() );
      self->setProperty( "errorLine", (int64) cfile->errorLine() );
   }

}

FALCON_FUNC  ConfParser_write( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();

   Item *i_stream = vm->param(0);

   bool bRes;

   if( i_stream == 0 )
   {
      bRes = cfile->save();
   }
   else {
      bool bValid = false;
      if ( i_stream->isObject() )
      {
         CoreObject *streamObj = i_stream->asObject();
         if ( streamObj->derivedFrom( "Stream" ) )
         {
            Stream *base = (Stream *) streamObj->getUserData();
            bRes = cfile->save( base );
            bValid = true;
         }
      }

      if ( ! bValid )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "Stream" ) ) );
         return;
      }
   }

   if ( ! bRes )
   {
      vm->raiseModError( new IoError( ErrorParam( 1260, __LINE__ ).
         desc( "Error parsing the file" ).extra( cfile->errorMessage() ) ) );
      self->setProperty( "error", cfile->errorMessage() );
      self->setProperty( "errorLine", (int64) cfile->errorLine() );
   }
}

FALCON_FUNC  ConfParser_get( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1);

   if ( i_key == 0 || ! i_key->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String value;

   if ( i_section != 0 && ! i_section->isNil() )
   {
      if ( ! cfile->getValue( *i_key->asString(), *i_section->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }
   else {
      if ( ! cfile->getValue( *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }

   // we have at least one value. but do we have more?
   String value1;
   if ( cfile->getNextValue( value1 ) )
   {
      CoreArray *array = new CoreArray( vm, 5 );
      array->append( new GarbageString( vm, value ) );
      array->append( new GarbageString( vm, value1 ) );

      while( cfile->getNextValue( value1 ) )
         array->append( new GarbageString( vm, value1 ) );

      vm->retval( array );
   }
   else {
      vm->retval( value );
   }
}


FALCON_FUNC  ConfParser_getOne( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1);

   if ( i_key == 0 || ! i_key->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String value;

   if ( i_section != 0 && ! i_section->isNil() )
   {
      if ( ! cfile->getValue( *i_key->asString(), *i_section->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }
   else {
      if ( ! cfile->getValue( *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }

   vm->retval( value );
}


FALCON_FUNC  ConfParser_getMultiple( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1);

   if ( i_key == 0 || ! i_key->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String value;
   if ( i_section != 0 && ! i_section->isNil() )
   {
      if ( ! cfile->getValue( *i_key->asString(), *i_section->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }
   else {
      if ( ! cfile->getValue( *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }

   CoreArray *array = new CoreArray( vm, 5 );
   array->append( new GarbageString( vm, value ) );

   String value1;
   while( cfile->getNextValue( value1 ) )
      array->append( new GarbageString( vm, value1 ) );

   vm->retval( array );
}


FALCON_FUNC  ConfParser_getSections( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();

   String section;
   CoreArray *ret = new CoreArray( vm );

   if( cfile->getFirstSection( section ) )
   {
      ret->append( new GarbageString( vm, section ) );
      while( cfile->getNextSection( section ) )
         ret->append( new GarbageString( vm, section ) );
   }

   vm->retval( ret );
}


FALCON_FUNC  ConfParser_getKeys( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param( 0 );

   if ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String key;
   CoreArray *ret = new CoreArray( vm );
   bool next;

   if ( i_section != 0 && ! i_section->isNil() ) {
      next = cfile->getFirstKey( *i_section->asString(), "", key );
   }
   else {
      next = cfile->getFirstKey( "", key );
   }

   while ( next )
   {
      ret->append( new GarbageString( vm, key ) );
      next = cfile->getNextKey( key );
   }

   vm->retval( ret );
}

FALCON_FUNC  ConfParser_getCategoryKeys( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_keyMask = vm->param( 0 );
   Item *i_section = vm->param( 1 );

   if ( i_keyMask == 0 || ! i_keyMask->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String key;
   CoreArray *ret = new CoreArray( vm );
   bool next;

   if ( i_section != 0 && ! i_section->isNil() ) {
      next = cfile->getFirstKey( *i_section->asString(), *i_keyMask->asString(), key );
   }
   else {
      next = cfile->getFirstKey( *i_keyMask->asString(), key );
   }

   while ( next )
   {
      ret->append( new GarbageString( vm, String( key, i_keyMask->asString()->length() + 1 ) ) );
      next = cfile->getNextKey( key );
   }

   vm->retval( ret );
}

FALCON_FUNC  ConfParser_getCategory( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_keyMask = vm->param( 0 );
   Item *i_section = vm->param( 1 );

   if ( i_keyMask == 0 || ! i_keyMask->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   if ( i_section != 0 && i_section->isNil() )
      i_section = 0;

   String key;
   CoreDict *ret = new LinearDict( vm );
   CoreDict *current = ret;
   bool next;

   if ( i_section != 0  ) {
      next = cfile->getFirstKey( *i_section->asString(), *i_keyMask->asString(), key );
   }
   else {
      next = cfile->getFirstKey( *i_keyMask->asString(), key );
   }

   while( next )
   {
      String value;

      // seeking a value won't alter key iterators.
      if( i_section != 0  )
         cfile->getValue( *i_section->asString(), key, value );
      else
         cfile->getValue( key, value );


      // we have at least one value. but do we have more?
      String value1;
      if ( cfile->getNextValue( value1 ) )
      {
         CoreArray *array = new CoreArray( vm, 5 );
         array->append( new GarbageString( vm, value ) );
         array->append( new GarbageString( vm, value1 ) );

         while( cfile->getNextValue( value1 ) )
            array->append( new GarbageString( vm, value1 ) );

         // we have used KEY; now what we want to save is just the non-category
         //current->addUnique( new GarbageString( vm, key, i_keyMask->asString()->length() + 1 ), array );
         current->insert( new GarbageString( vm, key), array );
      }
      else {
         //current->addUnique( new GarbageString( vm, key, i_keyMask->asString()->length() + 1 ), new GarbageString( vm, value ) );
         current->insert(  new GarbageString( vm, key) , new GarbageString( vm, value ) );
      }

      next = cfile->getNextKey( key );
   }

   vm->retval( ret );
}

FALCON_FUNC  ConfParser_getDictionary( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param( 0 );

   if ( i_section != 0 && ! i_section->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String key;
   CoreDict *ret = new LinearDict( vm );
   CoreDict *current = ret;
   bool next;

   if ( i_section != 0 ) {
      next = cfile->getFirstKey( *i_section->asString(), "", key );
   }
   else {
      next = cfile->getFirstKey( "", key );
   }

   while( next )
   {
      String value;

      // seeking a value won't alter key iterators.
      if( i_section != 0 )
         cfile->getValue( *i_section->asString(), key, value );
      else
         cfile->getValue( key, value );

      // we have at least one value. but do we have more?
      String value1;
      if ( cfile->getNextValue( value1 ) )
      {
         CoreArray *array = new CoreArray( vm, 5 );
         array->append( new GarbageString( vm, value ) );
         array->append( new GarbageString( vm, value1 ) );

         while( cfile->getNextValue( value1 ) )
            array->append( new GarbageString( vm, value1 ) );

         current->insert( new GarbageString( vm, key ), array );
      }
      else {
         current->insert( new GarbageString( vm, key ), new GarbageString( vm, value ) );
      }

      next = cfile->getNextKey( key );
   }

   vm->retval( ret );
}

/**
   ConfParser.add( key, value )
   ConfParser.add( section, key, value )

   Value may be anything, but it gets to-stringed.
*/
FALCON_FUNC  ConfParser_add( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_value = vm->param(1);
   Item *i_section = vm->param(2); // actually, if valorized, key and value are param 1 and 2.

   if ( i_key == 0 || ! i_key->isString() || i_value == 0 ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) ) );
      return;
   }

   String *value;
   bool delValue;
   if( i_value->isString() )
   {
      delValue = false;
      value = i_value->asString();
   }
   else {
      value = new GarbageString( vm );
      delValue = true;
      vm->itemToString( *value, i_value );
   }

   if( i_section == 0 || i_section->isNil() )
      cfile->addValue( *i_key->asString(), *value );
   else
      cfile->addValue( *i_section->asString(), *i_key->asString(), *value );

   if ( delValue )
      delete value;
}

FALCON_FUNC  ConfParser_set( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_value = vm->param(1);
   Item *i_section = vm->param(2); // actually, if valorized, key and value are param 1 and 2.

   if ( i_key == 0 || ! i_key->isString() || i_value == 0 ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) ) );
      return;
   }

   if ( i_section != 0 && i_section->isNil() )
      i_section = 0;

   String *value;
   bool delValue;

   if( i_value->isArray() )
   {
      CoreArray *array = i_value->asArray();
      bool first = true;

      for ( uint32 i = 0; i < array->length(); i ++ )
      {
         Item &itm = array->at( i );

         if( itm.isString() )
         {
            delValue = false;
            value = itm.asString();
         }
         else {
            value = new GarbageString( vm );
            delValue = true;
            vm->itemToString( *value, &itm );
         }

         if ( first )
         {
            // setValue will remove every previous reference...
            if( i_section == 0 )
               cfile->setValue( *i_key->asString(), *value );
            else
               cfile->setValue( *i_section->asString(), *i_key->asString(), *value );

            first = false;
         }
         else {
            // ...then we can begin to add
            if( i_section == 0 )
               cfile->addValue( *i_key->asString(), *value );
            else
               cfile->addValue( *i_section->asString(), *i_key->asString(), *value );
         }

         if ( delValue )
            delete value;
      }

      // we have no more business here
      return;
   }
   else if( i_value->isString() )
   {
      delValue = false;
      value = i_value->asString();
   }
   else {
      value = new GarbageString( vm );
      delValue = true;
      vm->itemToString( *value, i_value );
   }

   if( i_section == 0 )
      cfile->setValue( *i_key->asString(), *value );
   else
      cfile->setValue( *i_section->asString(), *i_key->asString(), *value );

   if ( delValue )
      delete value;
}

/**
   ConfParser.remove( key )
   ConfParser.remove( key, section )
*/

FALCON_FUNC  ConfParser_remove( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1); // optional

   if ( i_key == 0 || ! i_key->isString() ||
         ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) ) );
      return;
   }

   if ( i_section == 0 || i_section->isNil() )
   {
      cfile->removeValue( *i_key->asString() );
   }
   else
   {
      cfile->removeValue( *i_section->asString(), *i_key->asString() );
   }
}

/**
   ConfParser.removeCategory( key )
   ConfParser.removeCategory( key, section )
*/

FALCON_FUNC  ConfParser_removeCategory( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_category = vm->param(0);
   Item *i_section = vm->param(1); // optional

   if ( i_category == 0 || ! i_category->isString() ||
         ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) ) );
      return;
   }

   if ( i_section == 0 || i_section->isNil() )
   {
      cfile->removeCategory( *i_category->asString() );
   }
   else
   {
      cfile->removeCategory( *i_section->asString(), *i_category->asString() );
   }

}

FALCON_FUNC  ConfParser_addSection( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param(0);

   if ( i_section == 0 ||  ! i_section->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   vm->retval( (int64) ( cfile->addSection( *i_section->asString() ) == 0 ? 0: 1) );
}

FALCON_FUNC  ConfParser_removeSection( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param(0);

   if ( i_section == 0 ||  ! i_section->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   vm->retval( (int64) ( cfile->removeSection( *i_section->asString() ) ? 1: 0) );
}

FALCON_FUNC  ConfParser_clearMain( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   cfile->clearMainSection();
}

}
}

/* end of socket_ext.cpp */
