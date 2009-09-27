/*
   FALCON - The Falcon Programming Language
   FILE: json_ext.cpp

   JSON transport format interface - extension implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Sep 2009 18:28:44 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Funcext module main file - extension implementation.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/stream.h>
#include <falcon/stringstream.h>

#include "json_ext.h"
#include "json_mod.h"
#include "json_st.h"



/*#
   @beginmodule feather_funcext
*/

namespace Falcon {
namespace Ext {

/*#
   @function encode
   @brief Encode an item in JSON format.
   @param item the item to be encoded in JSON format.
   @optparam stream A stream on which to send the encoded result.
   @return a string containing the JSON string, if @b stream is nil
   @throw JSONError if the passed item cannot be turned into a JSON representation.
   @throw IoError in case of error on target stream .

   @code
      
   @endcode

   @note At the moment, the @b at function doesn't support BOM methods.
*/

FALCON_FUNC  json_encode ( ::Falcon::VMachine *vm )
{
   Item *i_item = vm->param(0);
   Item *i_stream = vm->param(1);

   Stream* target = 0;
   bool bDel;

   if ( i_item == 0 ||
      (i_stream != 0 && ! i_stream->isNil() && ! i_stream->isOfClass( "Stream" ))
        )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).
            extra("X, [Stream]") );
   }

   if ( i_stream == 0 || i_stream->isNil() )
   {
      bDel = true;
      target = new StringStream;
   }
   else {
      bDel = false;
      target = dyncast<Stream*>( i_stream->asObject()->getFalconData() );
   }

   JSON encoder;
   if( ! encoder.encode( *i_item, target ) )
   {
      throw new JSONError( ErrorParam( FALCON_JSON_NOT_CODEABLE, __LINE__  )
            .origin( e_orig_runtime )
            .desc( FAL_STR(json_msg_non_codeable) ) );
   }

   if( bDel )
   {
      vm->retval( static_cast<StringStream*>(target)->closeToString() );
      delete target;
   }
   else
   {
      if( ! target->good() )
      {
         throw new IoError(  ErrorParam( e_io_error, __LINE__ ).
            origin( e_orig_runtime ).
            sysError( target->lastError() ) );
      }
   }
}

FALCON_FUNC  json_decode ( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC  json_apply ( ::Falcon::VMachine *vm )
{
}

//=====================================================
// JSON Error
//
FALCON_FUNC  JSONError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new JSONError );

   ::Falcon::core::Error_init( vm );
}

}
}


/* end of funcext_ext.cpp */
