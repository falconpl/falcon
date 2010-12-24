/*
   FALCON - The Falcon Programming Language.
   FILE: webhelp.cpp

   Helpers for web applications
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Jun 2010 17:12:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

#include "core_module.h"
#include <falcon/base64.h>

namespace Falcon {
namespace core {

/*#
   @class Base64
   @brief Collection of methods to handle rfc3548 Base64 encoding.

*/

/*#
   @method encode Base64
   @brief Transforms an input string or MemBuf into a Base64 encoded string.
   @param data The data to be encoded.
   @return The encoded string.

   This @b static method encodes the contents of the incoming data as a
   @b base64 encoded string.

   The rfc3548 doesn't define the encoding of the input data, as base64 is a
   method to encode generic binary data and send them across the Internet.
   However, it's common practice to encode textual contents as utf-8 strings
   and then apply the base64 encoding.

   This method automatically uses utf-8
   encoding to transform strings with international characters. If this is
   not desired, provide a MemBuf as the parameter.

*/
FALCON_FUNC Base64_encode( VMachine* vm )
{
   Item* i_data = vm->param(0);
   if( i_data == 0 || ! (i_data->isMemBuf() || i_data->isString()) )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
            extra("S|M") );
   }

   CoreString* cs = new CoreString;

   if( i_data->isString() )
   {
      Base64::encode( *i_data->asString(), *cs );
   }
   else
   {
      Base64::encode( i_data->asMemBuf()->data(), i_data->asMemBuf()->size(), *cs );
   }

   vm->retval(cs);
}

/*#
   @method decode Base64
   @brief Decodes a previously encoded text data.
   @param data The data to be decoded.
   @raise ParseError if the incoming data is not a correct base64 string.
   @return The original string (as an international text).

   This @b static method decodes the contents of the incoming data as a
   @b base64 encoded string into a Falcon text-oriented String.

   The rfc3548 doesn't define the encoding of the input data, as base64 is a
   method to encode generic binary data and send them across the Internet.
   However, it's common practice to encode textual contents as utf-8 strings
   and then apply the base64 encoding.

   So, this method supposes that the data, to be transformed in a string,
   is actually an utf-8 representation of a text. If this is not desired,
   use the Base64.decmb method.
*/
FALCON_FUNC Base64_decode( VMachine* vm )
{
   Item* i_data = vm->param(0);
   if( i_data == 0 || ! i_data->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
            extra("S") );
   }

   CoreString* cs = new CoreString;
   if( ! Base64::decode( *i_data->asString(), *cs ) )
   {
      cs->mark(1);
      throw new ParseError( ErrorParam( e_parse_format, __LINE__ ) );
   }
   vm->retval(cs);
}

/*#
   @method decmb Base64
   @brief Decodes a previously encoded binary data.
   @param data The data to be decoded.
   @raise ParseError if the incoming data is not a correct base64 string.
   @return The origianl data, as a binary sequence of bytes.

   This @b static method decodes the contents of the incoming data as a
   @b base64 encoded string into a binary buffer.

*/
FALCON_FUNC Base64_decmb( VMachine* vm )
{
   Item* i_data = vm->param(0);
   if( i_data == 0 || ! i_data->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
            extra("S") );
   }

   const String& s = *i_data->asString();
   uint32 tgtsize = s.length() / 4*3+3;
   byte* tgt = (byte*) memAlloc( tgtsize );
   if ( ! Base64::decode( *i_data->asString(), tgt, tgtsize ) )
   {
      memFree( tgt );
      throw new ParseError( ErrorParam( e_parse_format, __LINE__ ) );
   }

   MemBuf_1* mb = new MemBuf_1( tgt, tgtsize, memFree );
   vm->retval(mb);
}

}
}

/* end of webhelp.cpp */
