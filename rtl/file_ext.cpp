/*
   FALCON - The Falcon Programming Language.
   FILE: file.cpp
   $Id: file_ext.cpp,v 1.15 2007/08/11 00:11:56 jonnymind Exp $

   File api
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun nov 1 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Falcon file api.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/cobject.h>
#include <falcon/fstream.h>
#include <falcon/sys.h>
#include <falcon/fassert.h>
#include <falcon/stdstreams.h>
#include <falcon/membuf.h>

namespace Falcon {
namespace Ext {

/** Closes a file. */
FALCON_FUNC  Stream_close ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
      vm->self().asObject()->getUserData() );

   if ( ! file->close() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError(  ErrorParam( 1110 ).origin( e_orig_runtime ).
            desc( "File error while closing the stream" ).sysError( (uint32) file->lastError()) ) );
      }
   }
}

/** Close a standard stream. */
FALCON_FUNC  StdStream_close ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Stream *file = static_cast<Stream *>( self->getUserData() );

   if ( file->close() )
   {
      if ( vm->hasProcessStreams() )
      {
         Item *mode = self->getProperty( "_stdStreamType" );
         if( mode != 0 && mode->isInteger() )
         {
            switch( mode->asInteger() )
            {
               case 0: vm->stdIn()->close(); break;
               case 1: vm->stdOut()->close(); break;
               case 2: vm->stdErr()->close(); break;
            }
         }
      }
   }
}

/** Reads from a file.
   read( size ) --> string
   read( string ) --> size
   read( membuf ) --> size
   read( string, size ) --> size
*/
FALCON_FUNC  Stream_read ( ::Falcon::VMachine *vm )
{
   MemBuf *membuf = 0;
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   Item *target = vm->param(0);
   String *cs_target;
   // if the third parameter is a not number, the second must be a string;
   // if the string is missing, we must create a new appropriate target.
   Item *last = vm->param(1);


   int32 size;
   bool returnTarget;

   if ( target == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( last != 0 ) {
      size = (int32) last->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      if ( target->isString() )
      {
         cs_target = target->asString();
         cs_target->reserve( size );
      }
      else {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            //"Given a size, the first parameter must be a string" );
         return;
      }
      
      returnTarget = false;
   }
   // we have only the second parameter.
   // it MUST be a string or an integer .
   else if ( target->isString() )
   {
      cs_target = target->asString();
      size = cs_target->allocated();
      if ( size <= 0 ) {
         size = cs_target->size();
         if ( size <= 0 ) {
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            return;
         }

         cs_target->bufferize(); // force to bufferize
      }
      returnTarget = false;
   }
   else if ( target->isMemBuf() )
   {
      cs_target = 0;
      membuf = target->asMemBuf();
      returnTarget = false;
   }
   else if ( target->isInteger() )
   {
      size = (int32) target->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }
      cs_target = new GarbageString( vm );
      cs_target->reserve( size );
      // no need to store for garbage, as we'll return this.
      returnTarget = true;
   }
   else
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         //"Second parameter must be either a string or an integer" );
      return;
   }

   if ( membuf != 0 )
      size = file->read( membuf->data(), membuf->size() );
   else
      size = file->read( cs_target->getRawStorage(), size );

   if ( size < 0 ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else if ( file->invalid() )
         vm->raiseModError( new IoError( ErrorParam( 1102 ).origin( e_orig_runtime ).
            desc( "Stream not open for reading" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1103 ).origin( e_orig_runtime ).
            desc( "File error while reading the stream" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   // valid also if size == 0
   if ( membuf == 0 )
      cs_target->size( size );

   if ( returnTarget ) {
      vm->retval( cs_target );
   }
   else {
      vm->retval((int64) size );
   }
}

/** Reads from a file.
   readText( size ) --> string
   readText( string ) --> size
   readText( string, size ) --> size
*/
FALCON_FUNC  Stream_readText ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   Item *target = vm->param(0);
   String *cs_target;
   // if the third parameter is a not number, the second must be a string;
   // if the string is missing, we must create a new appropriate target.
   Item *last = vm->param(1);

   int32 size;
   bool returnTarget;

   if ( target == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( last != 0 ) {
      size = (int32) last->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      if ( target->isString() )
      {
         cs_target = target->asString();
         cs_target->reserve( size );
      }
      else {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            //"Given a size, the first parameter must be a string" );
         return;
      }
      returnTarget = false;
   }
   // we have only the second parameter.
   // it MUST be a string or an integer .
   else if ( target->isString() )
   {
      cs_target = target->asString();
      size = cs_target->allocated();
      if ( size <= 0 ) {
         size = cs_target->size();
         if ( size <= 0 ) {
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            return;
         }

         cs_target->bufferize(); // force to bufferize
      }
      returnTarget = false;
   }
   else if ( target->isInteger() )
   {
      size = (int32) target->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }
      cs_target = new GarbageString( vm );
      cs_target->reserve( size );
      // no need to store for garbage, as we'll return this.
      returnTarget = true;
   }
   else
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         //"Second parameter must be either a string or an integer" );
      return;
   }

   if ( ! file->readString( *cs_target, size ) ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else if ( file->invalid() )
         vm->raiseModError( new IoError( ErrorParam( 1102 ).origin( e_orig_runtime ).
            desc( "Stream not open for reading" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1103 ).origin( e_orig_runtime ).
            desc( "File error while reading the stream" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   if ( returnTarget ) {
      vm->retval( cs_target );
   }
   else {
      vm->retval((int64) cs_target->length() );
   }
}


/** Reads a line from a file.
   readLine( size(=512) ) --> string
   readLine( string ) --> size
   readLine( string, size ) --> size
*/
FALCON_FUNC  Stream_readLine ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   if ( file == 0 ) return;
   Item *target = vm->param(0);
   String *cs_target;
   // if the third parameter is a not number, the second must be a string;
   // if the string is missing, we must create a new appropriate target.
   Item *last = vm->param(1);


   int32 size;
   bool returnTarget;

   if ( target == 0 ) {
      if ( last != 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            //"Given a size, the first parameter must be a string" );
         return;
      }
      size = 512;
      cs_target = new GarbageString( vm );
      cs_target->reserve( size );
      returnTarget = true;
   }
   else if ( last != 0 )
   {
      size = (int32) last->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      if ( target == 0 || target->type() == FLC_ITEM_STRING )
      {
         cs_target = target->asString();
         cs_target->size( 0 );
         // reserve a little size; it is ignored when there's enough space.
         cs_target->reserve( size );
      }
      else {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            //"Given a size, the first parameter must be a string" );
         return;
      }
      returnTarget = false;
   }
   // we have only the second parameter.
   // it MUST be a string or an integer .
   else if ( target->type() == FLC_ITEM_STRING )
   {
      cs_target = target->asString();
      size = cs_target->allocated();
      cs_target->size( 0 );

      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      cs_target->reserve( size ); // force to bufferize
      returnTarget = false;
   }
   else if ( target->type() == FLC_ITEM_INT ) {
      size = (int32) target->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }
      cs_target = new GarbageString( vm );
      cs_target->reserve( size );
      returnTarget = true;
   }
   else
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         //"Second parameter must be either a string or an integer" );
      return;
   }

   uint32 c = 0, c1 = 0;
   int pos = 0;
   bool getOk = file->get( c );
   while( getOk && pos < size )
   {
      if ( c == (uint32)'\r' ) {
         c1 = c;
         getOk = file->get( c );
         continue;
      }
      else if ( c == (uint32)'\n' ) {
         break;
      }
      else if ( c1 != 0 ) {
         c1 = 0;
         cs_target->append( c1 );
         ++pos;
      }

      cs_target->append( c );
      ++pos;
      getOk = file->get( c );
   }

   if ( ! getOk && ! file->eof() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else if ( file->invalid() )
         vm->raiseModError( new IoError( ErrorParam( 1102 ).origin( e_orig_runtime ).
            desc( "Stream not open for reading" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1103 ).origin( e_orig_runtime ).
            desc( "File error while reading the stream" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   if ( returnTarget ) {
      vm->retval( cs_target );
   }
   else {
      vm->retval( (int64) pos );
   }
}

/** Writes to a file.
   write( string ) --> size
   write( membuf, size [, start] ) --> size
   write( string, size [, start ] ) --> size
*/
FALCON_FUNC  Stream_write ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *source = vm->param(0);
   Item *count = vm->param(1);
   Item *i_start = vm->param(2);
   uint32 size, ssize, start;
   byte *buffer;

   if ( source == 0 || 
      ( ! source->isMemBuf() && ! source->isString() ) ||
      ( count != 0 && ! count->isOrdinal() ) ||
      ( i_start != 0 && ! i_start->isOrdinal() ))
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "S|M, [N, N]" ) ) );
      return;
   }

   if ( i_start == 0 )
      start = 0;
   else
      start = (uint32) i_start->forceInteger();

   if ( source->isMemBuf() )
   {
      MemBuf *mb = source->asMemBuf();
      buffer = mb->data();
      if ( count == 0 )
      {
         size = mb->size();
      }
      else {
         size = (uint32) count->forceInteger();
      }

      if ( size + start > mb->size() )
      {
         size = mb->size() - start; // can overflow...
      }

      //... but we'd return here, so it's ok
      if ( start > mb->size() || size == 0 )
      {
         // nothing to write
         vm->retval( 0 );
         return;
      }
   }
   else {
      ssize = (uint32) source->asString()->size();
      buffer = source->asString()->getRawStorage();
      if( count != 0 ) {
         size = (uint32) count->forceInteger();
         if ( size > ssize )
            size = ssize;
      }
      else
         size = ssize;
      
      if ( size + start > ssize )
      {
         size = ssize - start; // can overflow...
      }

      //... but we'd return here, so it's ok
      if ( start > ssize || size == 0 )
      {
         // nothing to write
         vm->retval( 0 );
         return;
      }

   }

   int64 written = file->write( buffer + start, size - start );
   if ( written < 0 )
   {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else if ( file->invalid() )
         vm->raiseModError( new IoError( ErrorParam( 1104 ).origin( e_orig_runtime ).
            desc( "Stream not open for writing" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1105 ).origin( e_orig_runtime ).
            desc( "File error while writing the stream" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   vm->retval( written );
}

/** Writes to a file.
   writeText( string )
   writeText( string, begin )
   writeText( string, begin, end )
*/
FALCON_FUNC  Stream_writeText ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *source = vm->param(0);
   Item *begin = vm->param(1);
   Item *end = vm->param(2);
   uint32 iBegin, iEnd;

   if ( source == 0 || source->type() != FLC_ITEM_STRING ||
      (begin != 0 && begin->type() != FLC_ITEM_INT ) ||
      (end != 0 && end->type() != FLC_ITEM_INT ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   iBegin = begin == 0 ? 0 : (uint32) begin->asInteger();
   iEnd = end == 0 ? source->asString()->length() : (uint32) end->asInteger();

   if ( ! file->writeString( *(source->asString()), iBegin, iEnd )  )
   {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else if ( file->invalid() )
         vm->raiseModError( new IoError( ErrorParam( 1104 ).origin( e_orig_runtime ).
            desc( "Stream not open for writing" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1105 ).origin( e_orig_runtime ).
            desc( "File error while writing the stream" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   vm->retval( (int64) 1 );
}


/** Seeks a position from the beginning of a file. */
FALCON_FUNC  Stream_seek ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *position = vm->param(0);
   if ( position== 0 || ! position->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int64 pos = file->seekBegin( position->forceInteger() );

   if ( file->bad() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1100 ).origin( e_orig_runtime ).
            desc( "Generic stream error" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   vm->retval( pos );
}

/** Seeks a position in a file. */
FALCON_FUNC  Stream_seekCur ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *position = vm->param(0);
   if ( position== 0 || ! position->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int64 pos = file->seekCurrent( position->forceInteger() );

   if ( file->bad() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1100 ).origin( e_orig_runtime ).
            desc( "Generic stream error" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   vm->retval( pos );
}


/** Seeks a position in a file. */
FALCON_FUNC  Stream_seekEnd ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *position = vm->param(0);
   if ( position== 0 || ! position->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int64 pos = file->seekEnd( position->forceInteger() );

   if ( file->bad() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1100 ).origin( e_orig_runtime ).
            desc( "Generic stream error" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }

   vm->retval( pos );
}

/** Return current position in a file. */
FALCON_FUNC  Stream_tell ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   int64 pos = file->tell();

   if ( file->bad() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1100 ).origin( e_orig_runtime ).
            desc( "Generic stream error" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }
   vm->retval( pos );
}

/** Truncate a file.
   truncate(); truncate at current position
   truncate( pos );  truncate at a given position
*/
FALCON_FUNC  Stream_truncate ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *position = vm->param(0);
   int64 pos;

   if ( position == 0 )
      pos = file->tell();
   else if ( ! position->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( pos == -1 || ! file->truncate( pos ) )
   {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError( ErrorParam( 1100 ).origin( e_orig_runtime ).
            desc( "Generic stream error" ).sysError( (uint32) file->lastError() ) ) );
      }
      return;
   }
}

/** Return last hard-error on the file. */
FALCON_FUNC  Stream_lastError ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( (int64) file->lastError() );
}

/** Return last hard-error on the file. */
FALCON_FUNC  Stream_lastMoved ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( (int64) file->lastMoved() );
}

/** Return true if at eof */
FALCON_FUNC  Stream_eof ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( file->eof() ? 1 : 0 );
}

/** Return true if open */
FALCON_FUNC  Stream_isOpen ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( file->open() ? 1 : 0 );
}

/** Return true if can read */
FALCON_FUNC  Stream_readAvailable ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *msecs_item = vm->param(0);
   int32 msecs = msecs_item == 0 ? 0 : (int32) msecs_item->forceInteger();


   int32 avail = file->readAvailable( msecs );
   if ( avail > 0 )
      vm->retval( 1 );
   else if ( avail == 0 )
      vm->retval( 0 );
   else if ( file->lastError() != 0 ) {
      vm->raiseModError( new IoError( ErrorParam( 1108 ).origin( e_orig_runtime ).
         desc( "Query on the stream failed" ).sysError( (uint32) file->lastError() ) ) );
      return;
   }
   else
      vm->retval( 0 );
}

/** Return true if can write */
FALCON_FUNC  Stream_writeAvailable ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *msecs_item = vm->param(0);
   int32 msecs = msecs_item == 0 ? 0 : (int32) msecs_item->forceInteger();

   if ( ! file->writeAvailable( msecs ) ) {
      if ( file->lastError() != 0 ) {
         vm->raiseModError( new IoError( ErrorParam( 1108 ).origin( e_orig_runtime ).
            desc( "Query on the stream failed" ).sysError( (uint32) file->lastError() ) ) );
         return;
      }
      vm->retval( 0 );
   }
   else
      vm->retval( 1 );
}

FALCON_FUNC  Stream_clone ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   // create a new stream instance.
   Item *clstream = vm->findWKI( "Stream" );
   fassert( clstream != 0 );
   CoreObject *obj = clstream->asClass()->createInstance();

   Stream *nstream = static_cast<Stream *>( file->clone() );
   // in case of filesystem error, we get 0 and system error properly set.
   if ( nstream == 0 )
   {
      // TODO: Raise uncloneable.
      vm->raiseModError( new IoError( ErrorParam( 1111 ).origin( e_orig_runtime ).
            desc( "Clone failed" ).sysError( (uint32) file->lastError() ) ) );
         return;
   }

   obj->setUserData( nstream );
   vm->retval( obj );
}


/** Return a representation of the last error */
FALCON_FUNC  Stream_errorDescription ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   String *str = new GarbageString( vm );
   file->errorDescription( *str );
   vm->retval( str );
}

/** Return a representation of the last error */
FALCON_FUNC  Stream_writeItem ( ::Falcon::VMachine *vm )
{
   CoreObject *fileObj = vm->self().asObject();
   Item *source = vm->param(0);

   if( source == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "X" ) ) );
      return;
   }

   Stream *file = static_cast<Stream *>( fileObj->getUserData() );
   Item::e_sercode sc = source->serialize( file, vm );
   switch( sc )
   {
      case Item::sc_ok: vm->retval( 1 ); break;
      case Item::sc_ferror: vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_runtime ) ) );
      default:
         vm->retnil(); // VM may already have raised an error.
   }
}

/** Return a representation of the last error */
FALCON_FUNC  Stream_readItem ( ::Falcon::VMachine *vm )
{
   // deserialize rises it's error if it belives it should.
   Stream *file = static_cast<Stream *>( vm->self().asObject()->getUserData() );
   Item::e_sercode sc = vm->regA().deserialize( file, vm );
   switch( sc )
   {
      case Item::sc_ok: return; // ok, we've nothing to do
      case Item::sc_ferror: vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_runtime ) ) );
      case Item::sc_misssym: vm->raiseModError( new GenericError( ErrorParam( e_undef_sym, __LINE__ ).origin( e_orig_runtime ) ) );
      case Item::sc_missclass: vm->raiseModError( new GenericError( ErrorParam( e_undef_sym, __LINE__ ).origin( e_orig_runtime ) ) );
      case Item::sc_invformat: vm->raiseModError( new GenericError( ErrorParam( e_invformat, __LINE__ ).origin( e_orig_runtime ) ) );

      case Item::sc_vmerror:
      default:
         vm->retnil(); // VM may already have raised an error.
         //TODO: repeat error.
   }
}

/** Opens a file.
   On success returns a stream;
   on failure throws an exception.
   Format: InputStream( name )
   Mode:
*/

FALCON_FUNC  InputStream_creator ( ::Falcon::VMachine *vm )
{
   Item *fileName = vm->param(0);
   if ( fileName == 0 || ! fileName->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Item *fileShare = vm->param(1);
   if ( fileShare != 0 && ! fileShare->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   ::Falcon::GenericStream::t_shareMode shMode = ::Falcon::GenericStream::e_smShareFull;
   if ( fileShare != 0 )
      shMode = (::Falcon::GenericStream::t_shareMode) fileShare->asInteger();

   FileStream *stream = new FileStream;
   stream->open( *fileName->asString(), ::Falcon::GenericStream::e_omReadOnly, shMode );

   if ( stream->lastError() != 0 )
   {
      vm->raiseModError( new IoError( ErrorParam( 1109 ).origin( e_orig_runtime ).
         desc( "Can't open file" ).extra(*fileName->asString()).sysError( (uint32) stream->lastError() ) ) );
      delete stream;
   }
   else {
      Item *stream_class = vm->findWKI( "Stream" );
      //if we wrote the std module, can't be zero.
      fassert( stream_class != 0 );

      ::Falcon::CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( stream );
      vm->retval( co );
   }
}


FALCON_FUNC  OutputStream_creator ( ::Falcon::VMachine *vm )
{
   Item *fileName = vm->param(0);
   if ( fileName == 0 || ! fileName->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Item *osMode = vm->param(1);
   int mode;

   if ( osMode == 0 ) {
      mode = 0666;
   }
   else
   {
      if ( ! osMode->isOrdinal() ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      mode = (int) osMode->forceInteger();
   }

   Item *fileShare = vm->param(2);
   if ( fileShare != 0 && ! fileShare->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   ::Falcon::GenericStream::t_shareMode shMode = ::Falcon::GenericStream::e_smShareFull;
   if ( fileShare != 0 )
      shMode = (::Falcon::GenericStream::t_shareMode ) fileShare->asInteger();

   FileStream *stream = new FileStream;
   stream->create( *fileName->asString(), (::Falcon::GenericStream::t_attributes) mode, shMode );

   if ( stream->lastError() != 0 )
   {
         vm->raiseModError( new IoError( ErrorParam( 1109 ).origin( e_orig_runtime ).
         desc( "Can't open file" ).sysError( (uint32) stream->lastError() ) ) );

      delete stream;
   }
   else {
      Item *stream_class = vm->findWKI( "Stream" );
      //if we wrote the std module, can't be zero.
      fassert( stream_class != 0 );
      ::Falcon::CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( stream );
      vm->retval( co );
   }
}

/** Opens or create a file.
   If the file doesn't exist, the function tries to create it.
   On success returns a stream;
   on failure throws an exception.
   Format: OutputStream( name, [os_mode] )
   Mode:
*/
FALCON_FUNC  IOStream_creator ( ::Falcon::VMachine *vm )
{
   Item *fileName = vm->param(0);
   if ( fileName == 0 || ! fileName->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Item *osMode = vm->param(1);
   int mode;

   if ( osMode == 0 ) {
      mode = 0666;
   }
   else
   {
      if ( ! osMode->isOrdinal() ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      mode = (int) osMode->forceInteger();
   }

   Item *fileShare = vm->param(2);
   if ( fileShare != 0 && ! fileShare->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   ::Falcon::GenericStream::t_shareMode shMode = ::Falcon::GenericStream::e_smShareFull;
   if ( fileShare != 0 )
      shMode = (::Falcon::GenericStream::t_shareMode ) fileShare->asInteger();

   FileStream *stream = new FileStream;
   stream->open( *fileName->asString(), ::Falcon::GenericStream::e_omReadWrite, shMode );

   if ( stream->lastError() != 0 )
   {
      stream->create( *fileName->asString(), (::Falcon::GenericStream::t_attributes) mode, shMode );
      if ( stream->lastError() != 0 )
      {
         vm->raiseModError( new IoError( ErrorParam( 1109 ).origin( e_orig_runtime ).
            desc( "Can't open file" ).extra(*fileName->asString()).sysError( (uint32) stream->lastError() ) ) );
         delete stream;
         return;
      }
   }

   Item *stream_class = vm->findWKI( "Stream" );
   //if we wrote the std module, can't be zero.
   fassert( stream_class != 0 );
   ::Falcon::CoreObject *co = stream_class->asClass()->createInstance();
   co->setUserData( stream );
   vm->retval( co );
}

static CoreObject *internal_make_stream( VMachine *vm, UserData *clone, int userMode )
{
   // The clone stream may be zero if the embedding application doesn't want
   // to share a virtual standard stream with us.
   if ( clone == 0 )
   {
       vm->raiseModError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime )  ) );
       return 0;
   }

   Item *stream_class;
   if ( userMode < 0 )
      stream_class = vm->findWKI( "Stream" );
   else
      stream_class = vm->findWKI( "StdStream" );

   //if we wrote the RTL module, can't be zero.
   fassert( stream_class != 0 );
   CoreObject *co = stream_class->asClass()->createInstance();
   co->setUserData( clone );
   if ( userMode >= 0 )
      co->setProperty( "_stdStreamType", userMode );

   vm->retval(co);
   return co;
}


FALCON_FUNC  _stdIn ( ::Falcon::VMachine *vm )
{
   if( vm->paramCount() == 0 )
   {
      internal_make_stream( vm, vm->stdIn()->clone(), -1 );
   }
   else {
      // verify streamability of parameter
      Item *p1 = vm->param(0);
      if( ! p1->isObject() || ! p1->asObject()->derivedFrom("Stream") )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      //keep the stream
      internal_make_stream( vm, vm->stdIn()->clone(), -1 ); // this also returns the old stream

      Stream *orig = (Stream *) p1->asObject()->getUserData();
      Stream *clone = (Stream *) orig->clone();
      if ( clone == 0 )
      {
         vm->raiseModError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime )  ) );
         return;
      }
      // but change it
      vm->stdIn( clone );
   }
}


FALCON_FUNC  _stdOut ( ::Falcon::VMachine *vm )
{
   if( vm->paramCount() == 0 )
   {
      internal_make_stream( vm, vm->stdOut()->clone(), -1 );
   }
   else {
      // verify streamability of parameter
      Item *p1 = vm->param(0);
      if( ! p1->isObject() || ! p1->asObject()->derivedFrom("Stream") )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      //keep the stream
      internal_make_stream( vm, vm->stdOut()->clone(), -1 );
      Stream *orig = (Stream *) p1->asObject()->getUserData();
      Stream *clone = (Stream *) orig->clone();
      if ( clone == 0 )
      {
         vm->raiseModError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime )  ) );
         return;
      }
      // but change it
      vm->stdOut( clone );
   }
}


FALCON_FUNC  _stdErr ( ::Falcon::VMachine *vm )
{
   if( vm->paramCount() == 0 )
   {
      internal_make_stream( vm, vm->stdErr()->clone(), -1 );
   }
   else {
      // verify streamability of parameter
      Item *p1 = vm->param(0);
      if( ! p1->isObject() || ! p1->asObject()->derivedFrom("Stream") )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }

      //keep the stream
      internal_make_stream( vm, vm->stdErr()->clone(), -1 );
      Stream *orig = (Stream *) p1->asObject()->getUserData();
      Stream *clone = (Stream *) orig->clone();
      if ( clone == 0 )
      {
         vm->raiseModError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime )  ) );
         return;
      }
      // but change it
      vm->stdErr( clone );
   }
}


FALCON_FUNC  stdInRaw ( ::Falcon::VMachine *vm )
{
   internal_make_stream( vm, new RawStdInStream(), 0 );
}

FALCON_FUNC  stdOutRaw ( ::Falcon::VMachine *vm )
{
   internal_make_stream( vm, new RawStdOutStream(), 1 );
}

FALCON_FUNC  stdErrRaw ( ::Falcon::VMachine *vm )
{
   internal_make_stream( vm, new RawStdErrStream(), 2 );
}


FALCON_FUNC  systemErrorDescription ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( ! number->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *str = new GarbageString( vm );
   ::Falcon::Sys::_describeError( number->forceInteger(), *str );
   vm->retval( str );
}


}}
/* end of file.cpp */
