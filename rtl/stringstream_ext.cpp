/*
   FALCON - The Falcon Programming Language.
   FILE: sstream.cpp

   Falcon module interface for string streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 5 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon module interface for string streams.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/stringstream.h>

namespace Falcon { namespace Ext {

/**
   String stream constructor.
   StringStream() --> stream
   StringStream( size ) --> stream
   StringStream( already_existing_string ) --> stream
*/
FALCON_FUNC  StringStream_init ( ::Falcon::VMachine *vm )
{
   // check the paramenter.
   Item *size_itm = vm->param( 0 );
   Stream *stream;

   if ( size_itm != 0 )
   {
      if ( size_itm->isString() ) {
         stream = new StringStream ( *size_itm->asString() );
      }
      else if ( size_itm->isOrdinal() )
      {
         stream = new StringStream ( (int32) size_itm->forceInteger() );
      }
      else
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }
   }
   else
      stream = new StringStream ();

   // get the self object
   CoreObject *self = vm->self().asObject();

   // create the string stream
   self->setUserData( stream );
}

/**
   StringStream.getString() --> string
*/
FALCON_FUNC  StringStream_getString ( ::Falcon::VMachine *vm )
{
   // get the self object
   CoreObject *self = vm->self().asObject();
   StringStream *ss = (StringStream *)self->getUserData();
   vm->retval( ss->getString() );
}

/**
   StringStream.closeToString() --> string
*/
FALCON_FUNC  StringStream_closeToString ( ::Falcon::VMachine *vm )
{
   // get the self object
   CoreObject *self = vm->self().asObject();
   StringStream *ss = (StringStream *)self->getUserData();
   GarbageString *rets = new GarbageString( vm );
   ss->closeToString( *rets );
   vm->retval( rets );
}

}}
/* end of sstream.cpp */
