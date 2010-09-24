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
#include <falcon/rosstream.h>

#include "json_ext.h"
#include "json_mod.h"
#include "json_st.h"



/*#
   @beginmodule feather_json
*/

namespace Falcon {
namespace Ext {


/*#
   @function JSONencode
   @brief Encode an item in JSON format.
   @param item the item to be encoded in JSON format.
   @optparam stream A stream on which to send the encoded result.
   @optparam pretty Add spacing around separators and puntaction.
   @optparam readable Put each item in lists on a separate line.
   @return a string containing the JSON string, if @b stream is nil
   @raise JSONError if the passed item cannot be turned into a JSON representation.
   @raise IoError in case of error on target stream.

*/

FALCON_FUNC  JSONencode ( ::Falcon::VMachine *vm )
{
   Item *i_item = vm->param(0);
   Item *i_stream = vm->param(1);
   Item *i_pretty = vm->param(2);
   Item *i_readable = vm->param(3);

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

   bool bPretty = i_pretty != 0 && i_pretty->isTrue();
   bool bReadable = i_readable != 0 && i_readable->isTrue();

   JSON encoder( bPretty, bReadable );
   bool result =  encoder.encode( *i_item, target );

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

   if ( ! result )
   {
      throw new JSONError( ErrorParam( FALCON_JSON_NOT_CODEABLE, __LINE__  )
            .origin( e_orig_runtime )
            .desc( FAL_STR(json_msg_non_codeable) ) );
   }

}

/*#
   @function JSONdecode
   @brief Decode an item stored in JSON format.
   @param source A string or a stream from which to read the JSON data.
   @return a string containing the JSON string, if @b stream is nil
   @raise JSONError if the input data cannot be parsed.
   @raise IoError in case of error on the source stream.

*/

FALCON_FUNC  JSONdecode ( ::Falcon::VMachine *vm )
{
   Item *i_source = vm->param(0);

   Stream* target = 0;
   bool bDel;

   if ( i_source == 0 || ! (i_source->isString() || i_source->isOfClass( "Stream" ))
        )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).
            extra("S|Stream") );
   }

   if ( i_source->isString() )
   {
      bDel = true;
      target = new ROStringStream(*i_source->asString());
   }
   else {
      bDel = false;
      target = dyncast<Stream*>( i_source->asObject()->getFalconData() );
   }

   Item item;
   JSON encoder;
   bool result = encoder.decode( item, target );

   // ok also in case of error -- actually better, as it clears garbage
   vm->retval( item );

   if( bDel )
   {
      delete target;
   }
   else if( ! target->good() && ! target->eof() )
   {
      throw new IoError(  ErrorParam( e_io_error, __LINE__ ).
         origin( e_orig_runtime ).
         sysError( target->lastError() ) );
   }

   if ( ! result )
   {
      throw new JSONError( ErrorParam( FALCON_JSON_NOT_DECODABLE, __LINE__  )
            .origin( e_orig_runtime )
            .desc( FAL_STR(json_msg_non_decodable) ) );
   }
}


//=====================================================
// JSON Error
//
/*#
   @class JSONError
   @brief Error generated after error conditions on JSON operations.
   @optparam code The error code
   @optparam desc The description for the error code
   @optparam extra Extra information specifying the error conditions.
   @from Error( code, desc, extra )
*/
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
