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
   uLong allocLen, compLen, currLen;
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

      vm->raiseModError( new GenericError( ErrorParam( err, __LINE__ )
                                         .desc( message ) ) );
      return;
   }

   String result;
   result.adopt( (char *) compData, compLen, allocLen );

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
   uLong allocLen, compLen, currLen;
   Bytef *compData;

   AutoCString asData( dataI->asString() );
   compLen = sizeof(char) * ( asData.length() * 200 );
   allocLen = compLen;

   compData = (Bytef *) memAlloc( compLen );

   err = uncompress( compData, &compLen, (const Bytef*) asData.c_str(), asData.length() );
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

      vm->raiseModError( new GenericError( ErrorParam( err, __LINE__ )
                                         .desc( message ) ) );
      return;
   }


   String result;
   result.adopt( (char *) compData, compLen, allocLen );

   vm->retval( result );
}

}
}

