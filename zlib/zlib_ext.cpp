/*
 * FALCON - The Falcon Programming Language
 * FILE: zlib_ext.cpp
 *
 * zlib module main file - extension definitions
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Thu Jan 3 2007
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <stdio.h>

#include <falcon/engine.h>

#include "zlib.h"
#include "zlib_ext.h"
#include "zlib_st.h"

/*#
   @beginmodule feather_zlib
*/

namespace Falcon {
namespace Ext {


static const String &internal_getErrorMsg( VMachine *vm, int error_code )
{
   switch ( error_code ) {
      case Z_MEM_ERROR:
         return FAL_STR(zl_msg_nomem);

      case Z_BUF_ERROR:
         return  FAL_STR(zl_msg_noroom);

      case Z_DATA_ERROR:
         return  FAL_STR( zl_msg_invformat );

      case Z_VERSION_ERROR:
         return  FAL_STR( zl_msg_vererr );
   }

   return  FAL_STR(zl_msg_generic);
}

/*#
   @class ZLib
   @brief ZLib encapsulation interface.

   Actually, this is a encapsulation class which is used to insolate  ZLib functions.
   Methods in this encapsulation are class-static methods of the ZLib class, and it
   is not necessary to create an instance of this class to use its methods.
*/

FALCON_FUNC ZLib_getVersion( ::Falcon::VMachine *vm )
{
   GarbageString *gsVersion = new GarbageString( vm, zlibVersion() );
   gsVersion->bufferize();
   vm->retval( gsVersion );
}

/*#
   @method compress ZLib
   @brief Compress a buffer (classwide method).
   @param buffer A string or a MemBuf to be compressed.
   @return A compressed buffer (in a byte-wide MemBuf).
   @raise ZLibError on compression error.

   This method will compress the data considering its raw memory value.
   This is suitable for bytewise strings loaded from binary streams
   and byte-wide memory buffers.

   Strings containing multi-byte characters can be compressed through
   this method, but the decompression process must know their original
   size and perform an adequate trancoding.

   For text strings, it is preferrable to use the @a ZLib.compressText
   function.
*/
FALCON_FUNC ZLib_compress( ::Falcon::VMachine *vm )
{
   Item *dataI = vm->param( 0 );
   if ( dataI == 0 || ( ! dataI->isString() && ! dataI->isMemBuf() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "S|M" ) ) );
      return;
   }

   int err;
   uLong allocLen, compLen, dataLen;
   Bytef *compData;
   const byte *data;

   if ( dataI->isString() )
   {
      data = dataI->asString()->getRawStorage();
      dataLen = dataI->asString()->size();
   }
   else {
      data = dataI->asMemBuf()->data();
      dataLen = dataI->asMemBuf()->size();
   }

   // for safety, reserve a bit more.
   compLen = dataLen < 512 ? dataLen *2 + 12: dataLen + 512;

   allocLen = compLen; // and keep a copy.
   compData = (Bytef *) memAlloc( compLen );

   do {
      err = compress( compData, &compLen, data, dataLen );

      // Buffer too small? -- try to enlarge it.
      if ( err == Z_BUF_ERROR )
      {
         memFree( compData );
         compLen += dataLen/2;
         allocLen = compLen;
         compData = (Bytef *) memAlloc( compLen );
      }
      else
         break;
   }
   while( true );

   if ( err != Z_OK )
   {
      // as err is < 0, we reverse it.
      vm->raiseModError( new ZLibError(
            ErrorParam( FALCON_ZLIB_ERROR_BASE - err, __LINE__ )
            .desc( internal_getErrorMsg( vm, err ) ) ) );
      return;
   }


   // eventually shrink a bit if we're using too much memory.
   if ( compLen < allocLen )
   {
      compData = (Bytef *) memRealloc( compData, compLen );
      allocLen = compLen;
   }

   MemBuf *result = new MemBuf_1( vm, compData, allocLen, true );
   vm->retval( result );
}


/*#
   @method compressText ZLib
   @brief Compress a text string (classwide method).
   @param text A string containing a text be compressed.
   @return A compressed buffer (in a byte-wide MemBuf).
   @raise ZLibError on compression error.

   This method will compress the a text so that an @a ZLib.uncompressText
   re-creates the original string.
*/
FALCON_FUNC ZLib_compressText( ::Falcon::VMachine *vm )
{
   Item *dataI = vm->param( 0 );
   if ( dataI == 0 || ! dataI->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra( "S" ) ) );
      return;
   }

   int err;
   uLong allocLen, compLen, dataLen;
   Bytef *compData;
   const byte *data;
   data = dataI->asString()->getRawStorage();
   dataLen = dataI->asString()->size();

   // for safety, reserve a bit more.
   compLen = dataLen < 512 ? dataLen *2 + 16: dataLen + 512;

   allocLen = compLen; // and keep a copy.
   compData = (Bytef *) memAlloc( compLen );
   compData[0] = (Bytef) dataI->asString()->manipulator()->charSize();
   compData[1] = (Bytef)(dataLen >> 24);
   compData[2] = (Bytef)((dataLen >> 16) & 0xff);
   compData[3] = (Bytef)((dataLen >> 8) & 0xff);
   compData[4] = (Bytef)((dataLen) & 0xff);
   compLen-=5;

   do {
      err = compress( compData+5, &compLen, data, dataLen );

      // Buffer too small? -- try to enlarge it.
      if ( err == Z_BUF_ERROR )
      {
         memFree( compData );
         compLen += dataLen/2;
         allocLen = compLen;
         compData[0] = (Bytef) dataI->asString()->manipulator()->charSize();
         compData[1] = (Bytef)(dataLen >> 24);
         compData[2] = (Bytef)((dataLen >> 16) & 0xff);
         compData[3] = (Bytef)((dataLen >> 8) & 0xff);
         compData[4] = (Bytef)((dataLen) & 0xff);
         compLen-=5;

         compData = (Bytef *) memAlloc( compLen );
      }
      else
         break;
   }
   while( true );

   if ( err != Z_OK )
   {
      vm->raiseModError( new ZLibError( ErrorParam( FALCON_ZLIB_ERROR_BASE -err, __LINE__ )
               .desc( internal_getErrorMsg( vm, err ) ) ) );
      return;
   }


   // eventually shrink a bit if we're using too much memory.
   if ( compLen + 5 < allocLen )
   {
      compData = (Bytef *) memRealloc( compData, compLen + 5 );
      allocLen = compLen + 5;
   }

   MemBuf *result = new MemBuf_1( vm, compData, allocLen, true );
   vm->retval( result );
}

/*#
   @method uncompress ZLib
   @brief Uncompress a buffer (classwide method).
   @param buffer A string or MemBuf containing previusly compressed data.
   @return A MemBuf containing the uncompressed data.
   @raise ZLibError on decompression error.

*/
FALCON_FUNC ZLib_uncompress( ::Falcon::VMachine *vm )
{
   Item *dataI = vm->param( 0 );

   if ( dataI == 0 || ( ! dataI->isString() && ! dataI->isMemBuf() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( "S|M" ) ) );
      return;
   }

   int err;
   uLong allocLen, compLen;
   Bytef *compData;
   const byte *dataIn;
   uint32 dataInSize;

   if ( dataI->isString() )
   {
      String *data = dataI->asString();
      dataIn = data->getRawStorage();
      dataInSize = data->size();
   }
   else {
      MemBuf *data = dataI->asMemBuf();
      dataIn = data->data();
      dataInSize = data->size();
   }

   // preallocate a good default memory
   compLen = sizeof(char) * ( dataInSize * 4 );
   if ( compLen < 512 )
   {
      compLen = 512;
   }

   allocLen = compLen;
   compData = (Bytef *) memAlloc( compLen );

   while( true )
   {
      err = uncompress( compData, &compLen, dataIn, dataInSize );

      if ( err == Z_MEM_ERROR )
      {
         //TODO: break also with Z_MEM_ERROR if we're using too much memory, like i.e. 512MB

         // try with a larger buffer
         compLen += dataInSize < 512 ? 512 : dataInSize * 4;
         allocLen = compLen;
         memFree( compData );
         compData = (Bytef *) memAlloc( compLen );
      }
      else
         break;
   }

   if ( err != Z_OK )
   {
      vm->raiseModError( new ZLibError(
               ErrorParam( FALCON_ZLIB_ERROR_BASE-err, __LINE__ )
               .desc( internal_getErrorMsg( vm, err ) ) ) );
      return;
   }

   if ( compLen < allocLen )
   {
      compData = (Bytef *) memRealloc( compData, compLen );
      allocLen = compLen;
   }

   MemBuf *result = new MemBuf_1( vm, compData, allocLen, true );
   vm->retval( result );
}

/*#
   @method uncompressText ZLib
   @brief Uncompress a buffer into a text (classwide method).
   @param buffer A MemBuf or string containing previusly compressed text data.
   @return A uncompressed string.
   @raise ZLibError on decompression error.

   The input @b buffer must be a string previously compressed
   with the @a ZLib.compressText method, or the function will fail.
*/
FALCON_FUNC ZLib_uncompressText( ::Falcon::VMachine *vm )
{
   Item *dataI = vm->param( 0 );

   if ( dataI == 0 || ( ! dataI->isString() && ! dataI->isMemBuf() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "S|M" ) ) );
      return;
   }

   int err;
   uLong compLen;
   Bytef *compData;
   const byte *dataIn;
   uint32 dataInSize;

   if ( dataI->isString() )
   {
      String *data = dataI->asString();
      dataIn = data->getRawStorage();
      dataInSize = data->size();
   }
   else {
      MemBuf *data = dataI->asMemBuf();
      dataIn = data->data();
      dataInSize = data->size();
   }

   // type of string
   if ( dataIn[0] != 1 && dataIn[0] != 2 && dataIn[0] != 4 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR(zl_msg_notct) ) ) );
      return;
   }

   // get length
   compLen = dataIn[1] << 24 | dataIn[2] << 16 | dataIn[3] << 8 | dataIn[4];
   compData = (Bytef *) memAlloc( compLen );

   err = uncompress( compData, &compLen, dataIn+5, dataInSize-5 );

   if ( err != Z_OK )
   {
      vm->raiseModError( new ZLibError(
               ErrorParam( FALCON_ZLIB_ERROR_BASE-err, __LINE__ )
               .desc( internal_getErrorMsg( vm, err ) ) ) );
      return;
   }

   GarbageString *result = new GarbageString( vm );
   result->adopt( (char *) compData, compLen, compLen );
   // set correct manipulator
   if (dataIn[0] == 2 )
      result->manipulator( &csh::handler_buffer16 );
   else if( dataIn[0] == 4 )
      result->manipulator( &csh::handler_buffer32 );

   vm->retval( result );
}


//=============================================================
// Zlib error
//

/*#
   @class ZLibError
   @brief Error generated by errors in the ZLib module.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   The value of the @b code property is set to one of the @a ZLibErrorCode
   values.

   See the Error class in the core module.
*/

/*#
   @init ZLibError
   @brief Initializes the zlib error.
*/

FALCON_FUNC  ZLibError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new ZLibError );

   ::Falcon::core::Error_init( vm );
}

}
}
