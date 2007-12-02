/*
   FALCON - The Falcon Programming Language.
   FILE: dict.cpp
   $Id: dict.cpp,v 1.8 2007/08/11 00:11:56 jonnymind Exp $

   Dictionary api
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio mar 16 2006
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
   Dictionary api
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/vm.h>
#include <falcon/fassert.h>
#include "rtl_messages.h"

namespace Falcon {

namespace Ext {

/****************************************
   Support for dictionaries
****************************************/

/*@begingroup dictsup Dictionary Support */



/*@function dictInsert
   @param dict a dictionary
   @param key the new key
   @param value the new value

   @short Inserts a new key/value pair in the dictionary

   This functions adds a couple of key/values in the dictionary,
   the key being forcibly unique.

   If the given key is already present is found on the dictionary,
   the previous value is overwritten.

*/

FALCON_FUNC  dictInsert ( ::Falcon::VMachine *vm )
{
   Item *dict = vm->param(0);
   Item *key = vm->param(1);
   Item *value = vm->param(2);


   if( dict == 0  || ! dict->isDict() || key == 0 || value == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   dict->asDict()->insert( *key, *value );
}

/*@function dictRemove
*/

FALCON_FUNC  dictRemove ( ::Falcon::VMachine *vm )
{
   Item *dict = vm->param(0);
   Item *key = vm->param(1);

   if( dict == 0  || ! dict->isDict() || key == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *d = dict->asDict();
   vm->retval( (int64) (d->remove( *key ) ? 1: 0) );
}

FALCON_FUNC  dictClear ( ::Falcon::VMachine *vm )
{
   Item *dict = vm->param(0);

   if( dict == 0  || ! dict->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *d = dict->asDict();
   d->clear();
}



/*@function dictMerge
   @param dict a dictionary
   @param other another dictionary

   @short Merges two dictionaries.
*/
FALCON_FUNC  dictMerge ( ::Falcon::VMachine *vm )
{
   Item *dict1 = vm->param(0);
   Item *dict2 = vm->param(1);
   if( dict1 == 0 || ! dict1->isDict() || dict2 == 0 || ! dict2->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *d1 = dict1->asDict();
   CoreDict *d2 = dict2->asDict();
   d1->merge( *d2 );
}


FALCON_FUNC  dictKeys( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);

   if( dict_itm == 0 || ! dict_itm->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   CoreArray *array = new CoreArray( vm );
   array->reserve( dict->length() );
   DictIterator *iter = dict->first();

   while( iter->isValid() )
   {
      array->append( iter->getCurrentKey() );
      iter->next();
   }
   delete iter;

   vm->retval( array );
}

/**
   Remember the optional param for extracting all the values of a key.
*/
FALCON_FUNC  dictValues( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);

   if( dict_itm == 0 || ! dict_itm->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   CoreArray *array = new CoreArray( vm );
   array->reserve( dict->length() );
   CoreIterator *iter = dict->first();

   while( iter->isValid() )
   {
      array->append( iter->getCurrent() );
      iter->next();
   }
   delete iter;

   vm->retval( array );
}


FALCON_FUNC  dictGet( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);
   Item *key_item = vm->param(1);

   if( dict_itm == 0 || ! dict_itm->isDict() || key_item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   Item *value = dict->find( *key_item );
   if ( value == 0 )
      vm->retnil();
   else
      vm->retval( *value );
}


FALCON_FUNC  dictFind( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);
   Item *key_item = vm->param(1);

   if( dict_itm == 0 || ! dict_itm->isDict() || key_item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // find the iterator class, we'll need it
   Item *i_iclass = vm->findGlobalItem( "Iterator" );
   if ( i_iclass == 0 || ! i_iclass->isClass() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_iterator_not_found ) ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();

   DictIterator *value = dict->findIterator( *key_item );
   if ( value == 0 )
      vm->retnil();
   else {
      CoreObject *ival = i_iclass->asClass()->createInstance();
      ival->setProperty( "_origin", *dict_itm );
      ival->setUserData( value );
      vm->retval( ival );
   }
}


FALCON_FUNC  dictBest( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);
   Item *key_item = vm->param(1);

   if( dict_itm == 0 || ! dict_itm->isDict() || key_item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // find the iterator class, we'll need it
   Item *i_iclass = vm->findGlobalItem( "Iterator" );
   if ( i_iclass == 0 || ! i_iclass->isClass() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_iterator_not_found ) ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   DictIterator *value = dict->first();
   CoreObject *ival = i_iclass->asClass()->createInstance();
   ival->setProperty( "_origin", *dict_itm );
   ival->setUserData( value );
   vm->regA() = ival;
   if ( ! dict->find( *key_item, *value ) )
   {
      vm->regA().setOob();
   }
}


/*@endgroup */

}
}

/* end of dict.cpp */
