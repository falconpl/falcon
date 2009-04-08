/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_service.h

   The SDL binding support module - intermodule services.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 25 Mar 2008 02:53:44 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL binding support module - intermodule services.
*/

#define FALCON_EXPORT_SERVICE

#include "sdl_service.h"
#include "sdl_mod.h"
#include <falcon/vm.h>
#include <falcon/stream.h>

namespace Falcon {

SDLService::SDLService():
   Service( SDL_SERVICE_SIGNATURE )
{
}

SDLService::~SDLService()
{
}

CoreObject *SDLService::createSurfaceInstance( VMachine *vm, ::SDL_Surface *surface )
{
   Item *cls = vm->findWKI( "SDLSurface" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance(surface);
   return obj;
}


//==========================================
// RWOps support for streams
//==========================================

extern "C" {

#define FALCON_SDLRWOPS_MAGIC 0xFA03238F

#define CHECK_SDL_RWOPS_MAGIC \
   if ( context->type != FALCON_SDLRWOPS_MAGIC )\
   {\
      SDL_SetError( "Invalid file type for fsdl_rwops" );\
      return -1;\
   }

inline Stream *getOpsStream( struct SDL_RWops* context )
{
   return (Stream *) context->hidden.unknown.data1;
}

static int fsdl_rwops_seek(struct SDL_RWops *context, int offset, int whence)
{
   CHECK_SDL_RWOPS_MAGIC;

   Stream *stream = getOpsStream( context );
   int64 pos = (int64) offset;
   int res;

   switch( whence )
   {
      case 0: res = (int) stream->seekBegin( pos ); break;
      case 1: res = (int) stream->seekCurrent( pos ); break;
      case 2: res = (int) stream->seekEnd( pos ); break;
      default:
         SDL_SetError( "Invalid whence parameter fsdl_rwops" );
         return -1;
   }

   if ( res == -1 ) {
      SDL_SetError( "Error in fsdl_rwops_seek" );
      return -1;
   }

   return res;
}

static int fsdl_rwops_read(struct SDL_RWops *context, void *ptr, int size, int maxnum)
{
   CHECK_SDL_RWOPS_MAGIC;

   Stream *stream = getOpsStream( context );

   int res = (int) stream->read( ptr, maxnum*size );
   if ( res == -1 ) {
      SDL_SetError( "Error in fsdl_rwops_read" );
      return -1;
   }

   return res;
}

static int fsdl_rwops_write(struct SDL_RWops *context, const void *ptr, int size, int num)
{
   CHECK_SDL_RWOPS_MAGIC;

   Stream *stream = getOpsStream( context );

   int res = (int) stream->write( ptr, num*size );
   if ( res == -1 ) {
      SDL_SetError( "Error in fsdl_rwops_write" );
      return -1;
   }

   return res;
}

static int fsdl_rwops_close(struct SDL_RWops *context)
{
   if ( context == 0 ) return 0; /* may be SDL_RWclose is called by atexit */

   CHECK_SDL_RWOPS_MAGIC;

   Stream *stream = getOpsStream( context );
   if ( ! stream->close() ) {
      SDL_SetError( "Error in fsdl_rwops_close" );
      return -1;
   }
   return 0;
}
}

void SDLService::rwopsFromStream( SDL_RWops &ops, Stream *stream )
{
   // types
   ops.type = FALCON_SDLRWOPS_MAGIC;

   // functions
   ops.seek = fsdl_rwops_seek;
   ops.write = fsdl_rwops_write;
   ops.read = fsdl_rwops_read;
   ops.close = fsdl_rwops_close;

   // data
   ops.hidden.unknown.data1 = stream;
}
}

/* end of sdl_service.cpp */
