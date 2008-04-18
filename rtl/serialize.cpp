/*
   FALCON - The Falcon Programming Language.
   FILE: serialize.cpp

   Serialization support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 13 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Serialization support
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/memory.h>

#include <falcon/stream.h>

namespace Falcon {
namespace Ext {


FALCON_FUNC  serialize ( ::Falcon::VMachine *vm )
{
   Item *fileId = vm->param(0);
   Item *source = vm->param(1);

   if( fileId == 0 || source == 0 || ! fileId->isObject() || ! fileId->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "X,O:Stream" ) ) );
      return;
   }


   Stream *file = (Stream *) fileId->asObject()->getUserData();
   Item::e_sercode sc = source->serialize( file );
   switch( sc )
   {
      case Item::sc_ok: vm->retval( 1 ); break;
      case Item::sc_ferror: vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_runtime ) ) );
      default:
         vm->retnil(); // VM may already have raised an error.
         //TODO: repeat error.
   }
}



FALCON_FUNC  deserialize ( ::Falcon::VMachine *vm )
{
   Item *fileId = vm->param(0);

   if( fileId == 0 || ! fileId->isObject() || ! fileId->isObject() || ! fileId->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "O:Stream" ) ) );
      return;
   }

   // deserialize rises it's error if it belives it should.
   Stream *file = (Stream *) fileId->asObject()->getUserData();
   Item::e_sercode sc = vm->regA().deserialize( file, vm );
   switch( sc )
   {
      case Item::sc_ok: return; // ok, we've nothing to do
      case Item::sc_ferror: vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_runtime ) ) );
      case Item::sc_misssym: vm->raiseModError( new GenericError( ErrorParam( e_undef_sym, __LINE__ ).origin( e_orig_runtime ) ) );
      case Item::sc_missclass: vm->raiseModError( new GenericError( ErrorParam( e_undef_sym, __LINE__ ).origin( e_orig_runtime ) ) );
      case Item::sc_invformat: vm->raiseModError( new GenericError( ErrorParam( e_invformat, __LINE__ ).origin( e_orig_runtime ) ) );

      case Item::sc_vmerror:
      default:
         vm->retnil(); // VM may already have raised an error.
         //TODO: repeat error.
   }
}


}}
/* end of serialize.cpp */
