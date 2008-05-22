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

/*#
   @beginmodule feather_zlib
*/

namespace Falcon {
namespace Ext {

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
   @raise @a ZLibError on compression error.

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
      String message;
      switch ( err ) {
      case Z_MEM_ERROR:
         message = "Not enough memory";
         break;
      }

      vm->raiseModError( new ZlibError( ErrorParam( -err, __LINE__ )
                                         .desc( message ) ) );
      return;
   }


   // eventually shrink a bit if we're using too much memory.
   if ( compLen < allocLen * 0.8 )
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
   @raise @a ZLibError on compression error.

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
   compData[0] = dataI->asString()->manipulator()->charSize();
   compData[1] = dataLen >> 24;
   compData[2] = (dataLen >> 16) & 0xff;
   compData[3] = (dataLen >> 8) & 0xff;
   compData[4] = (dataLen) & 0xff;
   compLen-=5;

   do {
      err = compress( compData+5, &compLen, data, dataLen );
      
      // Buffer too small? -- try to enlarge it.
      if ( err == Z_BUF_ERROR )
      {
         memFree( compData );
         compLen += dataLen/2;
         allocLen = compLen;
         compData[0] = dataI->asString()->manipulator()->charSize();
         compData[1] = dataLen >> 24;
         compData[2] = (dataLen >> 16) & 0xff;
         compData[3] = (dataLen >> 8) & 0xff;
         compData[4] = (dataLen) & 0xff;
         compLen-=5;

         compData = (Bytef *) memAlloc( compLen );
      }
      else
         break;
   }
   while( true );

   if ( err != Z_OK ) 
   {
      String message;
      switch ( err ) {
      case Z_MEM_ERROR:
         message = "Not enough memory";
         break;
      }

      vm->raiseModError( new ZlibError( ErrorParam( -err, __LINE__ )
                                         .desc( message ) ) );
      return;
   }


   // eventually shrink a bit if we're using too much memory.
   if ( compLen < allocLen * 0.8 )
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
   @raise @a ZLibError on compression error.

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
      String message;
      switch ( err ) {
      case Z_MEM_ERROR:
         message = "Not enough memory to uncompress";
         break;
      case Z_BUF_ERROR:
         message = "Not enough room in output buffer to decompress";
         break;
      case Z_DATA_ERROR:
         message = "Data supplied is not in compressed format";
         break;
      default:
         message = "An unknown uncompress error has occurred";
         break;
      }

      vm->raiseModError( new ZlibError( ErrorParam( -err, __LINE__ )
                                         .desc( message ) ) );
      return;
   }
   
   if ( compLen < allocLen / 2 )
   {
      compData = (Bytef *) memRealloc( compData, compLen );
      allocLen = compLen;
   }
   
   MemBuf *result = new MemBuf_1( vm, compData, allocLen, true );
   vm->retval( result );

   /*
   GarbageString *result = new GarbageString( vm );
   // eventually shrink a bit if we're using too much memory.
   if ( compLen < allocLen / 2 )
   {
      compData = (Bytef *) memRealloc( compData, compLen );
      allocLen = compLen;
   }

   result->adopt( (char *) compData, compLen, allocLen );
   vm->retval( result );
   */
}


//=============================================================
// Zlib error
//
FALCON_FUNC  ZlibError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new ZlibError ) );

   ::Falcon::core::Error_init( vm );
}

}
}
