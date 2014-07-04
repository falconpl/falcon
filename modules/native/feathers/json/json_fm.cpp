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

#define SRC "modules/native/feathers/json/json_fm.cpp"

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/stream.h>
#include <falcon/stringstream.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>

#include "json_fm.h"
#include "json_mod.h"

/*#
   @beginmodule json
*/

namespace Falcon {
namespace Feathers {


namespace CJSON {


/*#
   @method encode JSON
   @brief Encode an item in JSON format (static).
   @param item the item to be encoded in JSON format.
   @optparam stream A stream on which to send the encoded result.
   @optparam pretty Add spacing around separators and puntaction.
   @optparam readable Put each item in lists on a separate line.
   @return a string containing the JSON string, if @b stream is nil
   @raise JSONError if the passed item cannot be turned into a JSON representation.
   @raise IoError in case of error on target stream.

*/
FALCON_DECLARE_FUNCTION(encode, "item:X,stream:[Stream],pretty:[B],readable[B]")
FALCON_DEFINE_FUNCTION_P1(encode)
{
   static Class* cstream = Engine::instance()->stdHandlers()->streamClass();

   Item *i_item = ctx->param(0);
   Item *i_stream = ctx->param(1);
   Item *i_pretty = ctx->param(2);
   Item *i_readable = ctx->param(3);

   Stream* target = 0;
   bool bDel;

   if ( i_item == 0 ||
      (i_stream != 0 && ! i_stream->isNil() && ! i_stream->isInstanceOf( cstream ))
        )
   {
      throw paramError(__LINE__, SRC);
   }

   if ( i_stream == 0 || i_stream->isNil() )
   {
      bDel = true;
      target = new StringStream;
   }
   else {
      bDel = false;
      target = static_cast<Stream*>( i_stream->asInst() );
   }

   bool bPretty = i_pretty != 0 && i_pretty->isTrue();
   bool bReadable = i_readable != 0 && i_readable->isTrue();

   JSON encoder( target, bPretty, bReadable );

   if ( bDel )
   {
      // let the encoder to destroy the stream.
      target->decref();
   }

   String error;
   bool result =  encoder.encode( *i_item, error );

   if( bDel )
   {
      ctx->returnFrame( FALCON_GC_HANDLE(static_cast<StringStream*>(target)->closeToString()) );
   }
   else {
      ctx->returnFrame();
   }

   if ( ! result )
   {
      throw new JSONError( ErrorParam( FALCON_JSON_NOT_CODEABLE, __LINE__, SRC  )
            .desc( FALCON_JSON_NOT_CODEABLE_DESC )
            .extra(error) );
   }

}

/*#
   @method decode JSON
   @brief Decode an item stored in JSON format.
   @param source A string or a stream from which to read the JSON data.
   @return a string containing the JSON string, if @b stream is nil
   @raise JSONError if the input data cannot be parsed.
   @raise IoError in case of error on the source stream.

*/

FALCON_DECLARE_FUNCTION(decode, "source:[S|Stream]")
FALCON_DEFINE_FUNCTION_P1(decode)
{
   static Class* cstream = Engine::instance()->stdHandlers()->streamClass();
   Item *i_source = ctx->param(0);

   Stream* target = 0;
   bool bDel;

   if ( i_source == 0 || ! (i_source->isString() || i_source->isInstanceOf( cstream ))
        )
   {
      throw paramError();
   }

   if ( i_source->isString() )
   {
      bDel = true;
      target = new StringStream(*i_source->asString());
   }
   else {
      bDel = false;
      target = static_cast<Stream*>( i_source->asInst() );
   }

   Item item;
   JSON encoder( target );
   if( bDel )
   {
      // let the encoder to destroy the target
      target->decref();
   }

   String error;
   bool result = encoder.decode( item, error );

   if ( ! result )
   {
      throw new JSONError( ErrorParam( FALCON_JSON_NOT_DECODABLE, __LINE__, SRC  )
            .desc( FALCON_JSON_NOT_DECODABLE_DESC )
            .extra(error) );
   }

   ctx->returnFrame( item );
}

}


//====================================================================================================
// Class definition
//


ClassJSON::ClassJSON():
     Class("JSON")
{
   addMethod( new CJSON::Function_encode, true );
   addMethod( new CJSON::Function_decode, true );
}

ClassJSON::~ClassJSON()
{
}

int64 ClassJSON::occupiedMemory( void* ) const
{
   return 0;
}

void ClassJSON::dispose( void* ) const
{
   // do nothing; we're static
}

void* ClassJSON::clone( void* ) const
{
   // do nothing; we're static
   return 0;
}

void* ClassJSON::createInstance() const
{
   // do nothing.
   return 0;
}

//====================================================================================================
// Module
//

ModuleJSON::ModuleJSON():
         Module("json", true)
{
   addMantra(new ClassJSON );
   addMantra( new ClassJSONError );
}

ModuleJSON::~ModuleJSON()
{}

}
}


/* end of funcext_ext.cpp */
