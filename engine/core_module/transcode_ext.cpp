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

/*#
   @beginmodule core
*/

/** \file
   Transcoder api for rtl.
*/

#include <falcon/vm.h>
#include <falcon/stream.h>
#include <falcon/transcoding.h>
#include <falcon/coreobject.h>
#include <falcon/stdstreams.h>
#include <falcon/membuf.h>
#include "core_module.h"

namespace Falcon {
namespace core {

/*#
   @funset core_transcoding_functions Transcoding functions
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
   @brief Returns the "official" system encoding, if it matches with one known by Falcon.
   @return The system encoding name.

   This function will return the name under which Falcon knows the default
   system encoding. Using returned value, the program is able to create encoders
   that should be able to parse the data provided by system functions as directory
   scanning, or that is probably used as the main encoding for system related text
   files (i.e. configuration files).
*/

FALCON_FUNC  getSystemEncoding ( ::Falcon::VMachine *vm )
{
   CoreString *res = new CoreString;
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("S|M,[M]") );
   }

   CoreString *res = new CoreString;
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra( "S|M,S" ) );
   }

   CoreString *res = new CoreString;
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   vm->retval( res );
}

}
}

/* end of transcode_ext.cpp */
