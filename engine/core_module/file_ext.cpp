/*
   FALCON - The Falcon Programming Language.
   FILE: file.cpp

   File api
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun nov 1 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

/*#

*/

namespace Falcon {
namespace core {

/*#
   @class Stream
   @brief Stream oriented I/O class.
   @ingroup core_syssupport

   Stream class is a common interface for I/O operations. The class itself is to be
   considered “abstract”. It should never be directly instantiated, as factory
   functions, subclasses and embedding applications will provide fully readied
   stream objects.

   Stream I/O is synchronous, but it's possible to wait for the operation to be
   nonblocking with the readAvailable() and writeAvailable() methods.

   Generally, all the methods in the stream class raise an error in case of I/O
   failure.

   Streams provide also a character encoding layer; readText() and writeText() are
   meant to decode and encode falcon strings based on character encoding set with
   setEncoding(). Method as read() and write() are not affected, and seek
   operations works bytewise regardless the character conversion being used.

   @prop encoding Name of the set encoding, if given, for text operations
   @prop eolMode Mode of EOL conversion in text operations.
*/

/*#
   @method close Stream
   @brief Closes the stream.

   All the operations are flushes and system resources are freed.
   This method is also called automatically at garbage collection,
   if it has not been called before.
*/
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

/*#
   @method flush Stream
   @brief Flushes a stream.

   Ensures that the operations on the stream are correctly flushed.
*/
FALCON_FUNC  Stream_flush ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
      vm->self().asObject()->getUserData() );

   if ( ! file->flush() ) {
      if ( file->unsupported() )
         vm->raiseModError( new IoError( ErrorParam( 1101 ).origin( e_orig_runtime ).
            desc( "Unsupported operation for this file type" ) ) );
      else {
         vm->raiseModError( new IoError(  ErrorParam( 1110 ).origin( e_orig_runtime ).
            desc( "File error while flushing the stream" ).sysError( (uint32) file->lastError()) ) );
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
         Item mode;
         if( self->getProperty( "_stdStreamType", mode ) && mode.isInteger() )
         {
            switch( mode.asInteger() )
            {
               case 0: vm->stdIn()->close(); break;
               case 1: vm->stdOut()->close(); break;
               case 2: vm->stdErr()->close(); break;
            }
         }
      }
   }
}

/*#
   @method read Stream
   @brief Reads binary data from the stream.
   @param buffer A string or MemBuf that will be filled with read data.
   @optparam size Optionally, a maximum size to be read.
   @return Amount of data actually read (or a new string).
   @raise IoError on system errors.

   Read at maximum size bytes and returns the count of bytes that have
   been actually read. This version uses an already existing string
   as the destination buffer. If the string has not enough internal
   storage, it is reallocated to fit the required size. If the size
   parameter is not given, the internal storage size of the string is used
   as maximum read size; this is usually equal to len(buffer),
   but functions as @a strBuffer can create strings that have an internal
   storage wider than their length.

   The @b buffer parameter may be omitted; in that case, it is necessary to
   pass the @b size parameter. This method will then create a string wide
   enough to store the incoming data.
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

   if ( target == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 size = 0;
   bool returnTarget = false;

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

/*#
   @method readText Stream
   @brief Reads text encoded data from the stream.
   @param buffer A string that will be filled with read data.
   @optparam size Optionally, a maximum size to be read.
   @return Amount of data actually read (or a new string).
   @raise IoError on system errors.

   This method reads a string from a stream, eventually parsing the input
   data through the given character encoding set by the @a Stream.setEncoding method.
   The number of bytes actually read may vary depending on the decoding rules.

   If the size parameter is given, the function will try to read at maximum @b size
   characters, enlarging the buffer if there isn't enough room for the operation.
   If it is not given, the current allocated memory of buffer will be used instead.

   The @b buffer parameter may be omitted; in that case, it is necessary to
   pass the @b size parameter. This method will then create a string wide
   enough to store the incoming data.

   If the function is successful, the buffer may contain size characters or less
   if the stream hadn't enough characters to read.

   In case of failure, an IoError is raised.

   Notice that this function is meant to be used on streams that are known to have
   available all the required data. For streams that may perform partial
   updates (i.e. network streams), a combination of binary reads and
   @a transcodeFrom calls should be used instead.
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

/*#
   @method readLine Stream
   @brief Reads a line of text encoded data.
   @param buffer A string that will be filled with read data.
   @optparam size Maximum count of characters to be read before to return anyway.
   @return Amount of data actually read (or a new string).
   @raise IoError on system errors.

   This function works as @a Stream.readText, but if a new line is encountered,
   the read terminates. Returned string does not contain the EOL sequence.

   As for readText, this function may accept a numeric size as first parameter;
   in that case it will autonomously create a string wide enough to store
   the incoming data.
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

/*#
   @method write Stream
   @brief Write binary data to a stream.
   @param buffer A string or a MemBuf containing the data to be written.
   @optparam size Number of bytes to be written.
   @optparam start A position from where to start writing.
   @return Amout of data actually written.
   @raise IoError on system errors.

   Writes bytes from a buffer on the stream. The write operation is synchronous,
   and will block the VM until the stream has completed the write; however, the
   stream may complete only partially the operation. The number of bytes actually
   written on the stream is returned.

   If a size parameter is not given, it defaults to len( buffer ).

   A start position may optionally be given too; this allows to iterate through
   writes and send part of the data that couldent be send previously without
   extracting substrings or copying the memory buffers.
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
   else {
      start = (uint32) i_start->forceInteger();

      // minimal sanitization -- should we raise instead?
      if ( start < 0 )
      {
         start = 0;
      }
   }

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
      if ( start >= mb->size() || size == 0 )
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

/*#
   @method writeText Stream
   @brief Write text data to a stream.
   @param buffer A string containing the text to be written.
   @optparam start The character count from which to start writing data.
   @optparam end The position of the last character to write.
   @return Amout of characters actually written.
   @raise IoError on system errors.

   Writes a string to a stream using the character encoding set with
   @a Stream.setEncoding method. The begin and end optional parameters
   can be provided to write a part of a wide string without having to
   create a temporary substring.

   In case of failure, an IoError is raised.
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


/*#
   @method seek Stream
   @brief Moves the file pointer on seekable streams.
   @param position Position in the stream to seek.
   @return Position in the stream after seek.
   @raise IoError on system errors.

   Changes the position in the stream from which the next read/write operation
   will be performed. The position is relative from the start of the stream. If the
   stream does not support seeking, an IoError is raised; if the position is
   greater than the stream size, the pointer is set to the end of the file. On
   success, it returns the actual position in the file.
*/

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

/*#
   @method seekCur Stream
   @brief Moves the file pointer on seekable streams relative to current position.
   @param position Position in the stream to seek.
   @return Position in the stream after seek.
   @raise IoError on system errors.

   Changes the position in the stream from which the next read/write operation will
   be performed. The position is relative from the current position in the stream,
   a negative number meaning “backward”, and a positive meaning “forward”. If the
   stream does not support seeking, an IoError is raised. If the operation would
   move the pointer past the end of the file size, the pointer is set to the end;
   if it would move the pointer before the beginning, it is moved to the beginning.
   On success, the function returns the position where the pointer has been moved.
*/
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


/*#
   @method seekEnd Stream
   @brief Moves the file pointer on seekable streams relative to end of file.
   @param position Position in the stream to seek.
   @return Position in the stream after seek.
   @raise IoError on system errors.

   Changes the position in the stream from which the next read/write operation will
   be performed. The position is relative from the end of the stream. If the stream
   does not support seeking, an error is raised. If the operation would move the
   pointer before the beginning, the pointer is set to the file begin. On success,
   the function returns the position where the pointer has been moved. Use seekEnd(
   0 ) to move the pointer to the end of the stream.

   On success, the function returns the position where the pointer has been
   moved.
*/
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

/*#
   @method tell Stream
   @brief Return the current position in a stream.
   @return Position in the stream.
   @raise IoError on system errors.

   Returns the current read/write position in the stream. If the underlying
   stream does not support seeking, the operation raises an IoError.
*/
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

/*#
   @method truncate Stream
   @brief Resizes a file.
   @optparam position If given, truncate at given position.
   @return Position in the stream.
   @raise IoError on system errors.

   Truncate stream at current position, or if a position parameter is given,
   truncate the file at given position relative from file start. To empty a file,
   open it and then truncate it, or pass 0 as parameter.

   If the underlying stream does not support seek operation, this function will
   raise an error.
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
   else
      pos = position->forceInteger();

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

/*#
   @method lastError Stream
   @brief Return the last system error.
   @return An error code.

   Returns a system specific low level error code for last failed I/O
   operation on this stream, or zero if the last operation was succesful.
*/
FALCON_FUNC  Stream_lastError ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( (int64) file->lastError() );
}

/*#
   @method lastMoved Stream
   @brief Return the amount of data moved by the last operation.
   @return An amount of bytes.

   Returns the amount of bytes moved by the last write or read operation, in bytes.
   This may differ from the count of characters written or read by text-oriented
   functions.
*/
FALCON_FUNC  Stream_lastMoved ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( (int64) file->lastMoved() );
}

/*#
   @method eof Stream
   @brief Checks if the last read operation hit the end of the file.
   @return True if file pointer is at EOF.

   Returns true if the last read operation hit the end of file.

*/
FALCON_FUNC  Stream_eof ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( file->eof() ? 1 : 0 );
}

/*#
   @method isOpen Stream
   @brief Checks if the stream is currently open.
   @return True if the file is open.

   Return true if the stream is currently open.
*/
FALCON_FUNC  Stream_isOpen ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   vm->retval( file->open() ? 1 : 0 );
}

/*#
   @method readAvailable Stream
   @brief Checks if data can be read, or wait for available data.
   @optparam seconds Maximum wait in seconds and fraction (defaults to infinite).
   @return True if data is available, false otherwise.
   @raise IoError On stream error.
   @raise InterruptedError if the Virtual Machine is asynchronously interrupted.

   This function checks if data is available on a stream to be read immediately,
   or if it becomes available during a determined time period. The @b seconds
   parameter may be set to zero to perform just a check, or to a positive value
   to wait for incoming data. If the parameter is not given, or if it's set to
   a negative value, the wait will be infinite.

   A read after readAvailable has returned succesfully is granted not to be
   blocking (unless another coroutine or thread reads data from the same stream
   in the meanwhile). Performing a read after that readAvailable has returned
   false will probably block for an undefined amount of time.

   This method complies with the @a interrupt_protocol of the Virtual Machine.
*/
FALCON_FUNC  Stream_readAvailable ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *secs_item = vm->param(0);
   int32 msecs = secs_item == 0 ? -1 : (int32) (secs_item->forceNumeric()*1000);


   int32 avail = file->readAvailable( msecs, &vm->systemData() );
   if ( file->interrupted() )
   {
      vm->interrupted( true, true, true );
      return;
   }

   if ( avail > 0 )
      vm->regA().setBoolean( true );
   else if ( avail == 0 )
      vm->regA().setBoolean( false );
   else if ( file->lastError() != 0 ) {
      vm->raiseModError( new IoError( ErrorParam( 1108 ).origin( e_orig_runtime ).
         desc( "Query on the stream failed" ).sysError( (uint32) file->lastError() ) ) );
      return;
   }
   else
      vm->regA().setBoolean( false );
}

/*#
   @method writeAvailable Stream
   @brief Checks if data can be written, or wait until it's possible to write.
   @optparam seconds Maximum wait in seconds and fraction (defaults to infinite).
   @return True if data can be written, false otherwise.
   @raise IoError On stream error.
   @raise InterruptedError if the Virtual Machine is asynchronously interrupted.

   This function checks if the stream is available for immediate write,
   or if it becomes available during a determined time period. The @b seconds
   parameter may be set to zero to perform just a check, or to a positive value
   to wait for the line being cleared. If the @b seconds is not given, or if it's set to
   a negative value, the wait will be infinite.

   A write operation after writeAvailable has returned succesfully is granted not to be
   blocking (unless another coroutine or thread writes data to the same stream
   in the meanwhile). Performing a read after that readAvailable has returned
   false will probably block for an undefined amount of time.

   This method complies with the @a interrupt_protocol of the Virtual Machine.
*/
FALCON_FUNC  Stream_writeAvailable ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );

   Item *secs_item = vm->param(0);
   int32 msecs = secs_item == 0 ? -1 : (int32) (secs_item->forceNumeric()*1000);

   if ( file->writeAvailable( msecs, &vm->systemData() ) <= 0 )
   {
      if ( file->interrupted() )
      {
         vm->interrupted( true, true, true );
         return;
      }

      if ( file->lastError() != 0 ) {
         vm->raiseModError( new IoError( ErrorParam( 1108 ).origin( e_orig_runtime ).
            desc( "Query on the stream failed" ).sysError( (uint32) file->lastError() ) ) );
         return;
      }
      vm->retval( 0 );
   }
   else {
      vm->retval( 1 );
   }
}

/*#
   @method clone Stream
   @brief Clone the stream handle.
   @return A new copy of the stream handle.
   @raise IoError On stream error.

   The resulting stream is interchangeable with the original one. From this point
   on, write and read operations are not reflected on the cloned object, so two
   stream objects can be effectively used to read and write at different places in
   the same resource, unless the underlying stream is not seekable (in which case,
   reads are destructive).

   The underlying resource remains open until all the instances of the streams are
   closed.
*/
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


/*#
   @method errorDescription Stream
   @brief Returns a system specific textual description of the last error.
   @return A string describing the system error.

   Returns a system specific textual description of the last error occurred on the stream.
*/
FALCON_FUNC  Stream_errorDescription ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getUserData() );
   String *str = new GarbageString( vm );
   file->errorDescription( *str );
   vm->retval( str );
}

/*#
   @method writeItem Stream
   @brief Serializes an item to the stream.
   @param item The item to be serialized.
   @raise IoError On stream error.

   Serializes an item to the given stream. This method works as @b serialize, but
   it works as a method of this stream instead of being considered a function.
*/

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
   Item::e_sercode sc = source->serialize( file );
   switch( sc )
   {
      case Item::sc_ok: vm->retval( 1 ); break;
      case Item::sc_ferror: vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_runtime ) ) );
      default:
         vm->retnil(); // VM may already have raised an error.
   }
}

/*#
   @method readItem Stream
   @brief Deserializes an item from the stream.
   @return The deserialized item.
   @raise IoError on underlying stream error.
   @raise GenericError If the data is correctly de-serialized, but it refers to
         external symbols non defined by this script.
   @raise ParseError if the format of the input data is invalid.

   Deerializes an item from the given stream. This method works as @b deserialize, but
   it works as a method of this stream instead of being considered a function.
*/
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
      case Item::sc_invformat: vm->raiseModError( new ParseError( ErrorParam( e_invformat, __LINE__ ).origin( e_orig_runtime ) ) );

      case Item::sc_vmerror:
      default:
         vm->retnil(); // VM may already have raised an error.
         //TODO: repeat error.
   }
}

/*#
   @funset rtl_stream_factory Stream factory functions
   @brief Function creating or opening file and system streams.

   Stream factory functions create a stream object bound to a system file.
   As the open or create operation may fail, using a factory function is
   more appropriate, as they can avoid creating the underlying stream object
   when not needed. Also, they can setup the proper stream subclass to manage
   special files.

   Stream factory function often accept a parameter to determine the sharing mode.
   Only some systems implements correctly this functionality, so use with caution.
   Predefined constant that can be used as share mode are:

   - @b FILE_EXCLUSIVE: The calling scripts own the shared file; other processes
        trying to open the file, or even the same process, should receive an error.
   - @b FILE_SHARE_READ: A file opened in this way can be opened by other processes,
         but only for read operations.
   - @b FILE_SHARE: Any process may open and overwrite the file.

   Function creating files, as IOStream() and OutputStream()
   accepts also a creation ownership mode. This is actually an octal
   number that is directly passed to the POSIX systems for directory ownership
   creation. It has currently no meaning for MS-Windows systems.

   @beginset core_stream_factory
*/

/*#
   @function InputStream
   @brief Open a system file for reading.
   @ingroup core_syssupport
   @param fileName A relative or absolute path to a file to be opened for input
   @optparam shareMode If given, the share mode.
   @return A new valid @a Stream instance on success.

   If the function is successful, it opens the given fileName and returns a
   valid stream object by which the underlying file may be read. Calling write
   methods on the returned object will fail, raising an error.

   If the optional share parameter is not given, the maximum share publicity
   available on the system will be used.

   If the file cannot be open, an error containing a valid fsError code is raised.

   See @a rtl_stream_factory for a description of the shared modes.
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

/*#
   @function OutputStream
   @brief Creates a stream for output only.
   @ingroup core_syssupport
   @param fileName A relative or absolute path to a file to be opened for input
   @optparam createMode If given, the ownership of the created file.
   @optparam shareMode If given, the share mode.
   @return A new valid @a Stream instance on success.

   If the function is successful, it creates the given fileName and returns a valid
   stream object by which the underlying file may be read. If an already existing
   file name is given, then the file is truncated and its access right are updated.
   Calling read methods on the returned object will fail, raising an error.

   If the file can be created, its sharing mode can be controlled by providing a
   shareMode parameter. In case the shareMode is not given, then the maximum
   publicity is used.

   If the file cannot be created, an error containing a valid fsError code is
   raised.

   See @a rtl_stream_factory for a description of the shared modes.
*/
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

/*#
   @function IOStream
   @brief Creates a stream for input and output.
   @ingroup core_syssupport
   @param fileName A relative or absolute path to a file to be opened for input
   @optparam createMode If given, the ownership of the created file.
   @optparam shareMode If given, the share mode.
   @return A new valid @a Stream instance on success.

   If the function is successful, it creates the given fileName and returns a valid
   stream object by which the underlying file may be read. If an already existing
   file name is given, then the file is truncated and its access right are updated.
   Calling read methods on the returned object will fail, raising an error.

   If the file can be created, its sharing mode can be controlled by providing a
   shareMode parameter. In case the shareMode is not given, then the maximum
   publicity is used.

   If the file cannot be created, an error containing a valid fsError code is
   raised.

   See @a rtl_stream_factory for a description of the shared modes.
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

static CoreObject *internal_make_stream( VMachine *vm, FalconData *clone, int userMode )
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

/*#
   @function stdIn
   @brief Creates an object mapped to the standard input of the Virtual Machine.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned read-only stream is mapped to the standard input of the virtual
   machine hosting the script. Read operations will return the characters from the
   input stream as they are available. The readAvailable() method of the returned
   stream will indicate if read operations may block. Calling the read() method
   will block until some character can be read, or will fill the given buffer up
   the amount of currently available characters.

   The returned stream is a clone of the stream used by the Virtual Machine as
   standard input stream. This means that every transcoding applied by the VM is
   also available to the script, and that, when running in embedding applications,
   the stream will be handled by the embedder.

   As a clone of this stream is held in the VM, closing it will have actually no
   effect, except that of invalidating the instance returned by this function.

   Read operations will fail raising an I/O error.
*/
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

/*#
   @function stdOut
   @brief Creates an object mapped to the standard output of the Virtual Machine.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps output operations on the standard output stream of
   the process hosting the script.

   The returned stream is a clone of the stream used by the Virtual Machine as
   standard output stream. This means that every transcoding applied by the VM is
   also available to the script, and that, when running in embedding applications,
   the stream will be handled by the embedder.

   As a clone of this stream is held in the VM, closing it will have actually no
   effect, except that of invalidating the instance returned by this function.

   Read operations will fail raising an I/O error.
*/

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

/*#
   @function stdErr
   @brief Creates an object mapped to the standard error of the Virtual Machine.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps output operations on the standard error stream of
   the virtual machine hosting the script.

   The returned stream is a clone of the stream used by the Virtual Machine as
   standard error stream. This means that every transcoding applied by the VM is
   also available to the script, and that, when running in embedding applications,
   the stream will be handled by the embedder.

   As a clone of this stream is held in the VM, closing it will have actually no
   effect, except that of invalidating the instance returned by this function.

   Read operations will fail raising an I/O error.
*/
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

/*#
   @function stdInRaw
   @brief Creates a stream that interfaces the standard input stream of the host process.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps input operations on the standard input of the
   process hosting the script. The returned stream is bound directly with the
   process input stream, without any automatic transcoding applied.
   @a Stream.readText will read the text as stream of binary data coming from the
   stream, unless @a Stream.setEncoding is explicitly called on the returned
   instance.

   Closing this stream has the effect to close the standard input stream of the
   process running the script (if the operation is allowed by the embedding
   application).  Applications trying to write data to the script process will be
   notified that the script has closed the stream and is not willing to receive
   data anymore.

   The stream is read only. Write operations will cause an I/O to be raised.
*/
FALCON_FUNC  stdInRaw ( ::Falcon::VMachine *vm )
{
   internal_make_stream( vm, new RawStdInStream(), 0 );
}

/*#
   @function stdOutRaw
   @brief Creates a stream that interfaces the standard output stream of the host process.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps output operations on the standard output stream of the
   process hosting the script. The returned stream is bound directly with the
   process output, without any automatic transcoding applied. @a Stream.writeText
   will write the text as stream of bytes to the stream, unless
   @a Stream.setEncoding is explicitly called on the returned instance.

   Closing this stream has the effect to close the standard output of the process
   running the script (if the operation is allowed by the embedding application).
   Print functions, fast print operations, default error reporting and so on will
   be unavailable from this point on.

   Applications reading from the output stream of the process running the scripts,
   in example, piped applications, will recognize that the script has completed
   its output, and will disconnect immediately, while the script may continue to run.

   The stream is write only. Write operations will cause an I/O to be raised.
*/

FALCON_FUNC  stdOutRaw ( ::Falcon::VMachine *vm )
{
   internal_make_stream( vm, new RawStdOutStream(), 1 );
}

/*#
   @function stdErrRaw
   @brief Creates a stream that interfaces the standard error stream of the host process.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps output operations on the standard error stream of the
   process hosting the script. The returned stream is bound directly with the
   process error stream, without any automatic transcoding applied.
   @a Stream.writeText will write the text as stream of bytes to the stream,
   unless @a Stream.setEncoding is explicitly called on the returned
   instance.

   Closing this stream has the effect to close the standard error stream of the
   process running the script (if the operation is allowed by the embedding
   application).  Applications reading from the error stream of the script will be
   notified that the stream has been closed, and won't be left pending in reading
   this stream.

   The stream is write only. Read operations will cause an I/O to be raised.
*/
FALCON_FUNC  stdErrRaw ( ::Falcon::VMachine *vm )
{
   internal_make_stream( vm, new RawStdErrStream(), 2 );
}

/*#
   @function systemErrorDescription
   @inset core_general_purpose
   @brief Returns a system dependent message explaining an integer error code.
   @param errorCode A (possibly) numeric error code that some system function has returned.
   @return A system-specific error description.

   This function is meant to provide the users (and the developers) with a
   minimal support to get an hint on why some system function failed, without
   having to consult the system manual pages. The fsError field of the Error class
   can be fed directly inside this function.
*/

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

/*#
   @function fileCopy
   @ingroup core_syssupport
   @param source Source file to be copied
   @param dest Destination file.
   @brief Copies a whole file from one position to another.
   @raise IoError on system error.

   This function performs a file copy. The function is still
   experimental and needs addition of VM interruption protocol
   compliancy, as well as the possibility to preserve or change
   the system attributes in the target copy.
*/
FALCON_FUNC  fileCopy ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   Item *filedest = vm->param(1);

   if ( filename == 0 || ! filename->isString() ||
        filedest == 0 || ! filedest->isString()
      )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra("S,S") ) );
      return;
   }

   const String &source = *filename->asString();
   const String &dest = *filedest->asString();

   ::Falcon::GenericStream::t_shareMode shMode = ::Falcon::GenericStream::e_smShareFull;

   FileStream instream, outstream;
   instream.open( source, ::Falcon::GenericStream::e_omReadOnly, shMode );
   if ( ! instream.good() )
   {
      vm->raiseModError( new IoError( ErrorParam( e_io_error, __LINE__ ).
         extra( source ).
         sysError( (uint32) instream.lastError() ) ) );
      return;
   }

   outstream.create( dest, (Falcon::GenericStream::t_attributes) 0644, shMode );
   if ( ! outstream.good() )
   {
      instream.close();
      vm->raiseModError( new IoError( ErrorParam( e_io_error, __LINE__ ).
         extra( dest ).
         sysError( (uint32) outstream.lastError() ) ) );
      return;
   }

   byte buffer[4096];
   int count = 0;
   while( ( count = instream.read( buffer, 4096) ) > 0 )
   {
      if ( outstream.write( buffer, count ) < 0 )
      {
         vm->raiseModError( new IoError( ErrorParam( e_io_error, __LINE__ ).
            sysError( (uint32) outstream.lastError() ) ) );
         instream.close();
         outstream.close();
         return;
      }
   }

   if ( count < 0 )
   {
      vm->raiseModError( new IoError( ErrorParam( e_io_error, __LINE__ ).
            sysError( (uint32) instream.lastError() ) ) );
   }

   instream.close();
   outstream.close();
}


}}
/* end of file.cpp */
