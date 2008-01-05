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
 * In order to use this file in it's compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes bundled with this
 * package.
 */

#include <stdio.h>

#include <falcon/engine.h>

#include "zlib.h"
#include "zlib_ext.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC ZLib_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   GarbageString *gsVersion = new GarbageString( vm, zlibVersion() );
   self->setProperty( "version", gsVersion );

   vm->retval( self );
}

FALCON_FUNC ZLib_compress( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *dataI = vm->param( 0 );
   if ( dataI == 0 || ! dataI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
   }

   int err;
   uLong allocLen, compLen;
   Bytef *compData;

   AutoCString asData( dataI->asString() );

   compLen = sizeof(char) * asData.length();
   compLen += (int) (asData.length() * 0.3);
   compLen += 12;

   allocLen = compLen;

   compData = (Bytef *) memAlloc( compLen );

   err = compress( compData, &compLen, (const Bytef*) asData.c_str(), asData.length() );
   if ( err != Z_OK ) {
      String message;
      switch ( err ) {
      case Z_MEM_ERROR:
         message = "Not enough memory";
         break;

      case Z_BUF_ERROR:
         message = "Not enough room in the output buffer";
         break;
      }

      vm->raiseModError( new ZlibError( ErrorParam( -err, __LINE__ )
                                         .desc( message ) ) );
      return;
   }

   GarbageString *result = new GarbageString( vm );
   // eventually shrink a bit if we're using too much memory.
   if ( compLen < allocLen / 2 )
   {
      compData = (Bytef *) memRealloc( compData, compLen );
      allocLen = compLen;
   }

   result->adopt( (char *) compData, compLen, allocLen );
   vm->retval( result );
}


FALCON_FUNC ZLib_uncompress( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *dataI = vm->param( 0 );
   if ( dataI == 0 || ! dataI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
   }

   int err;
   uLong allocLen, compLen;
   Bytef *compData;
   String *data = dataI->asString();

   // preallocate a good default memory
   compLen = sizeof(char) * ( data->size() * 4 );
   if ( compLen < 512 )
   {
      compLen = 512;
   }

   allocLen = compLen;
   compData = (Bytef *) memAlloc( compLen );

   while( true )
   {
      err = uncompress( compData, &compLen, data->getRawStorage(), data->size() );

      if ( err == Z_MEM_ERROR )
      {
         //TODO: break also with Z_MEM_ERROR if we're using too much memory, like i.e. 512MB

         // try with a larger buffer
         compLen += data->size() < 512 ? 512 : data->size() * 4;
         allocLen = compLen;
         memFree( compData );
         compData = (Bytef *) memAlloc( compLen );
      }
      else
         break;
   }

   if ( err != Z_OK ) {
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

   GarbageString *result = new GarbageString( vm );
   // eventually shrink a bit if we're using too much memory.
   if ( compLen < allocLen / 2 )
   {
      compData = (Bytef *) memRealloc( compData, compLen );
      allocLen = compLen;
   }

   result->adopt( (char *) compData, compLen, allocLen );
   vm->retval( result );
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

