/*
   FALCON - The Falcon Programming Language
   FILE: transcode_ext.cpp

   Transcoder api for rtl.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 2 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Transcoder api for rtl.
*/

#include <falcon/vm.h>
#include <falcon/stream.h>
#include <falcon/transcoding.h>
#include <falcon/cobject.h>
#include <falcon/stdstreams.h>
#include <falcon/membuf.h>
#include "core_module.h"

/*#
   
*/

namespace Falcon {

/*#
   @method setEncoding Stream
   @brief Set the text encoding and EOL mode for text-based operations.
   @param encoding Name of the encoding that is used for the stream.
   @optparam EOLMode How to treat end of line indicators.

   This method sets an encoding that will affect readText() and writeText() methods.
   Provided encodings are:
   - "utf-8"
   - "utf-16"
   - "utf-16BE"
   - "utf-16LE"
   - "iso8859-1" to "iso8859-15"
   - "cp1252"
   - "C" (byte oriented – writes byte per byte)

   As EOL manipulation is also part of the text operations, this function allows to
   chose how to deal with EOL characters stored in Falcon strings when writing data
   and how to parse incoming EOL. Available values are:
   - CR_TO_CR: CR and LF characters are untranslated
   - CR_TO_CRLF: When writing, CR (“\n”) is translated into CRLF, when reading CRLF is translated into a single “\n”
   - SYSTEM_DETECT: use host system policy.

   If not provided, this parameter defaults to SYSTEM_DETECT.

   If the given encoding is unknown, a ParamError is raised.
*/

FALCON_FUNC  Stream_setEncoding ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Stream *file = reinterpret_cast<Stream *>( self->getUserData() );

   Item *i_encoding = vm->param(0);
   Item *i_eolMode = vm->param(1);

   if ( i_encoding == 0 || ! i_encoding->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int mode = ( i_eolMode == 0 ? SYSTEM_DETECT : (int) i_eolMode->forceInteger());
   if( mode != SYSTEM_DETECT && mode != CR_TO_CR && mode != CR_TO_CRLF )
   {
      mode = SYSTEM_DETECT;
   }

   Transcoder *trans = TranscoderFactory( *(i_encoding->asString()), file, true );

   if ( trans == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Stream *final;
   if ( mode == SYSTEM_DETECT )
   {
      final = AddSystemEOL( trans );
   }
   else if( mode == CR_TO_CRLF )
   {
      final = new TranscoderEOL( trans, true );
   }
   else
      final = trans;

   self->setUserData( final );
   self->setProperty( "encoding", *i_encoding );
   self->setProperty( "eolMode", (int64) mode );
}

/*#
   @funset rtl_transcoding_functions Transcoding functions
   @brief Functions needed to transcode texts into various character sets.

   Transcoding functions turns binary strings encoded in a format into
   Falcon strings, or conversely, they turn Falcon strings into binary
   encoded buffers. Used in combination with binary stream reads and write,
   this function allows full internationalization of script input and output.

   However, if the target stream is known to support safe reads and writes and
   to provide immediate access to the needed data, the @a Stream.setEncoding method
   is more efficient, as it doesn't need a temporary buffer to store the binary read
   data, or the binary data that has to be written.

   @beginset core_transcoding_functions

*/

/*#
   @function getSystemEncoding
   @brief Returns the “official” system encoding, if it matches with one known by Falcon.
   @return The system encoding name.

   This function will return the name under which Falcon knows the default
   system encoding. Using returned value, the program is able to create encoders
   that should be able to parse the data provided by system functions as directory
   scanning, or that is probably used as the main encoding for system related text
   files (i.e. configuration files).
*/

FALCON_FUNC  getSystemEncoding ( ::Falcon::VMachine *vm )
{
   String *res = new GarbageString( vm );
   GetSystemEncoding( *res );
   vm->retval( res );
}

/*#
   @function transcodeTo
   @brief Returns a binary buffer containing an encoded representation of a Falcon string.
   @param string Falcon string to be encoded.
   @param encoding Name of the encoding (as a string).
   @return On success, the transcoded string.
   @raise ParamError if the encoding is not known.

   In case the encoding name is not known, the function will raise a ParamError.
*/
FALCON_FUNC  transcodeTo ( ::Falcon::VMachine *vm )
{
   Item *i_source = vm->param( 0 );
   Item *i_encoding = vm->param( 1 );

   if ( i_source == 0 || ( ! i_source->isString() && ! i_source->isMemBuf() ) ||
        i_encoding == 0 || ! i_encoding->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *res = new GarbageString( vm );
   String *source;
   String dummy;

   if ( i_source->isMemBuf() )
   {
      source = &dummy;
      // using 0 as allocated, the buffer is considered static.
      dummy.adopt( (char *) i_source->asMemBuf()->data(), i_source->asMemBuf()->size(), 0 );
   }
   else
   {
      source = i_source->asString();
   }

   if ( ! TranscodeString( *source, *(i_encoding->asString()), *res ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( res );
}

/*#
   @function transcodeFrom
   @brief Returns a Falcon string created by parsing the given one as a binary sequence of bytes.
   @param string Falcon string or MemBuf to be encoded.
   @param encoding Name of the encoding (as a string).
   @return On success, the transcoded string.
   @raise ParamError if the encoding is not known.

   In case the encoding name is not known, the function will raise a ParamError.
   The transcoding may also fail if the source data is not a valid sequence under the
   given encoding, and cannot be decoded.
*/
FALCON_FUNC  transcodeFrom ( ::Falcon::VMachine *vm )
{
   Item *i_source = vm->param( 0 );
   Item *i_encoding = vm->param( 1 );

   if ( i_source == 0 || ( ! i_source->isString() && !i_source->isMemBuf() ) ||
        i_encoding == 0 || ! i_encoding->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *res = new GarbageString( vm );
   String *source;
   String dummy;

   if ( i_source->isMemBuf() )
   {
      source = &dummy;
      // using 0 as allocated, the buffer is considered static.
      dummy.adopt( (char *) i_source->asMemBuf()->data(), i_source->asMemBuf()->size(), 0 );
   }
   else
   {
      source = i_source->asString();
   }

   if ( ! TranscodeFromString( *source, *(i_encoding->asString()), *res ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( res );
}
}

/* end of transcode_ext.cpp */
