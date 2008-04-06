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
#include "falcon_rtl_ext.h"

namespace Falcon {

/** Add a stream transcoder.
   Stream_setTranscoding( encoding, eolTranscoder );
   \param encoding the target or source encoding
   \param eolTranscoder  nil = system detect, CR_to_CRLF, CR_to_CRLF or SYSTEM_DETECT
   Mode:
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


FALCON_FUNC  getSystemEncoding ( ::Falcon::VMachine *vm )
{
   String *res = new GarbageString( vm );
   GetSystemEncoding( *res );
   vm->retval( res );
}

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
