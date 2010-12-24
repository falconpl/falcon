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

/*#
   @beginmodule core
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/coreobject.h>
#include <falcon/fstream.h>
#include <falcon/sys.h>
#include <falcon/fassert.h>
#include <falcon/stdstreams.h>
#include <falcon/membuf.h>
#include <falcon/streambuffer.h>
#include <falcon/uri.h>
#include <falcon/vfsprovider.h>
#include <falcon/transcoding.h>

#include "core_module.h"


namespace Falcon {
namespace core {

// raises te correct error depending on the problem on the file
static void s_breakage( Stream *file )
{
   if ( file->unsupported() )
      throw new IoError( ErrorParam( e_io_unsup )
         .origin( e_orig_runtime ) );

   else if ( file->invalid() )
      throw new IoError( ErrorParam( e_io_invalid )
         .origin( e_orig_runtime ) );

   throw new IoError( ErrorParam( e_io_error )
         .origin( e_orig_runtime )
         .sysError( (uint32) file->lastError() ) );

}

/*#
   @class Stream
   @brief Stream oriented I/O class.
   @ingroup core_syssupport

   Stream class is a common interface for I/O operations. The class itself is to be
   considered "abstract". It should never be directly instantiated, as factory
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
   Stream *file = static_cast<Stream *>( vm->self().asObject()->getFalconData() );

   // declaring the VM idle from now on.
   VMachine::Pauser pauser( vm );

   if ( ! file->close() )
   {
      s_breakage( file );
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
      vm->self().asObject()->getFalconData() );

   // declaring the VM idle from now on.
   VMachine::Pauser pauser( vm );

   if ( ! file->flush() )
   {
      s_breakage( file );
   }
}

/** Close a standard stream. */
FALCON_FUNC  StdStream_close ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Stream *file = static_cast<Stream *>( self->getFalconData() );

   // declaring the VM idle from now on.
   VMachine::Pauser pauser( vm );

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
   @return Amount of data actually read.
   @raise IoError on system errors.

   This method uses an already existing and pre-allocated
   string or Memory Buffer, filling it with at maximum @b size bytes. If
   @b size is not provided, the method tries to read enough data to fill
   the given buffer. A string may be pre-allocated with the @a strBuffer
   function.

   If @b size is provided but it's larger than the available space in
   the given buffer, it is ignored. If there isn't any available space in
   the target buffer, a ParamError is raised.

   If the buffer is a string, each read fills the string from the beginning.
   If it is a MemBuffer, the space between @a MemoryBuffer.limit and @a len is
   filled. This allow for partial reads in slow (i.e. network) streams.
*/
FALCON_FUNC  Stream_read ( ::Falcon::VMachine *vm )
{
   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );

   Item *i_target = vm->param(0);
   Item *i_size = vm->param(1);

   if ( i_target == 0 || ! ( i_target->isString() || i_target->isMemBuf() )
      || ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "S|M,[N]" ) );
   }

   int32 size;
   byte *memory;

   // first, extract the maximum possible size
   if ( i_target->isString() )
   {
      String *str = i_target->asString();
      if ( str->allocated() == 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->moduleString( rtl_string_empty ) ) );
      }

      memory = str->getRawStorage();
      size = str->allocated();
   }
   else
   {
      MemBuf* mb = i_target->asMemBuf();
      size = mb->remaining();

      if( size <= 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->moduleString( rtl_buffer_full ) ) );
      }

      memory = mb->data() + (mb->position() * mb->wordSize());
   }

   if( i_size != 0 )
   {
      int32 nsize = (int32) i_size->forceInteger();

      if( nsize <= 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->moduleString( rtl_zero_size ) ) );
      }

      if ( nsize < size )
         size = nsize;
   }

   // delcare the VM idle during the I/O
   vm->idle();
   size = file->read( memory, size );
   vm->unidle();

   if ( size < 0 )
   {
      s_breakage( file );
   }

   if ( i_target->isString() )
   {
      i_target->asString()->size( size );
   }
   else
   {
      MemBuf* mb = i_target->asMemBuf();
      mb->position( i_target->asMemBuf()->position() + size );
      fassert( mb->limit() <= mb->length() );
   }

   vm->retval((int64) size );
}


/*#
   @method grab Stream
   @brief Grabs binary data from the stream.
   @param size Maximum size to be read.
   @return A string containing binary data from the stream.
   @raise IoError on system errors.

   This metod creates a string wide enough to read size bytes,
   and then tries to fill it with binary data coming from the stream.
*/
FALCON_FUNC  Stream_grab ( ::Falcon::VMachine *vm )
{
   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );
   Item *i_size = vm->param(0);

   if ( i_size == 0 || ! i_size->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "N" ) );
   }

   int32 nsize = (int32) i_size->forceInteger();

   if( nsize <= 0 )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->moduleString( rtl_zero_size ) ) );
   }

   CoreString* str = new CoreString;
   str->reserve( nsize );

   vm->idle();
   int32 size = file->read( str->getRawStorage(), nsize );
   vm->unidle();

   if ( size < 0 )
   {
      s_breakage( file );
   }


   str->size( size );
   vm->retval( str );
}


/*#
   @method readText Stream
   @brief Reads text encoded data from the stream.
   @param buffer A string that will be filled with read data.
   @optparam size Optionally, a maximum size to be read.
   @return Amount of data actually read.
   @raise IoError on system errors.

   This method reads a string from a stream, eventually parsing the input
   data through the given character encoding set by the @a Stream.setEncoding method.
   The number of bytes actually read may vary depending on the decoding rules.

   If the size parameter is given, the function will try to read at maximum @b size
   characters, enlarging the string if there isn't enough room for the operation.
   If it is not given, the current allocated memory of buffer will be used instead.

   @note This differ from Stream.read, where the buffer is never grown, even when
   it is a string.

   If the function is successful, the buffer may contain @b size characters or less
   if the stream hadn't enough characters to read.

   In case of failure, an IoError is raised.

   Notice that this function is meant to be used on streams that are known to have
   available all the required data. For streams that may perform partial
   updates (i.e. network streams), a combination of binary reads and
   @a transcodeFrom calls should be used instead.
*/

FALCON_FUNC  Stream_readText ( ::Falcon::VMachine *vm )
{
   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );
   Item *i_target = vm->param(0);
   Item *i_size = vm->param(1);

   if ( i_target == 0 || ! i_target->isString()
      || ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "S,[N]" ) );
   }

   String *str = i_target->asString();
   int32 size;

   if ( i_size != 0  )
   {
      size = (int32) i_size->forceInteger();
      if ( size <= 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->moduleString( rtl_zero_size ) ) );
         return;
      }
      str->reserve( size );
   }
   else
   {
      size = str->allocated();

      if ( size <= 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->moduleString( rtl_string_empty ) ) );
         return;
      }
   }

   vm->idle();
   if ( ! file->readString( *str, size ) )
   {
      vm->unidle();
      s_breakage( file );
   }
   else {
      vm->unidle();
   }

   vm->retval( (int64) str->length() );
}

/*#
   @method grabText Stream
   @brief Grabs text encoded data from the stream.
   @param size Optionally, a maximum size to be read.
   @return A string containing the read text.
   @raise IoError on system errors.

   This method reads a string from a stream, eventually parsing the input
   data through the given character encoding set by the @a Stream.setEncoding method,
   and returns it in a newly allocated string.

   If the function is successful, the buffer may contain @b size characters or less
   if the stream hadn't enough characters to read.

   In case of failure, an IoError is raised.

   Notice that this function is meant to be used on streams that are known to have
   available all the required data. For streams that may perform partial
   updates (i.e. network streams), a combination of binary reads and
   @a transcodeFrom calls should be used instead.
*/

FALCON_FUNC  Stream_grabText ( ::Falcon::VMachine *vm )
{
   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );
   Item *i_size = vm->param(0);

   if ( i_size == 0 || ! i_size->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "N" ) );
   }

   CoreString *str = new CoreString;
   int32 size = (int32) i_size->forceInteger();
   if ( size <= 0 )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->moduleString( rtl_zero_size ) ) );
      return;
   }
   str->reserve( size );

   vm->idle();
   if ( ! file->readString( *str, size ) )
   {
      vm->unidle();
      s_breakage( file ); // will throw
   }
   else {
      vm->unidle();
   }

   vm->retval( str );
}

/*#
   @method readLine Stream
   @brief Reads a line of text encoded data.
   @param buffer A string that will be filled with read data.
   @optparam size Maximum count of characters to be read before to return anyway.
   @return True if a line was read, false otherwise.
   @raise IoError on system errors.

   This function works as @a Stream.readText, but if a new line is encountered,
   the read terminates. Returned string does not contain the EOL sequence.
   Also, the returned string may be empty if the line was empty.
   
   At EOF, the function returns false. Example:

   @code
   s = InputStream( "file.txt" )
   line = strBuffer(4096)
   while s.readLine( line ): > "LINE: ", line
   s.close()
   @endcode
   
   @note It is possible to obtain a newly allocated line instead of having to
   provide a target buffer through the @a Stream.grabLine method.
   
   @see Stream.grabLine
*/
FALCON_FUNC  Stream_readLine ( ::Falcon::VMachine *vm )
{
   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );
   Item *i_target = vm->param(0);
   Item *i_size = vm->param(1);

   if ( i_target == 0 || ! i_target->isString()
      || ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "S,[N]" ) );
   }

   String *str = i_target->asString();
   int32 size;

   if ( i_size != 0  )
   {
      size = (int32) i_size->forceInteger();
      if ( size <= 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->moduleString( rtl_zero_size ) ) );
         return;
      }
      str->reserve( size );
   }
   else
   {
      size = str->allocated();

      if ( size <= 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->moduleString( rtl_string_empty ) ) );
         return;
      }
   }

   // anyhow, reset the string.
   str->size(0);

   // well, if we're eof before starting, then we can't read.
   if ( file->eof() )
   {
	  vm->regA().setBoolean(false);
	  return;
   }

   // we're idling while using VM memory, but we know we're keeping the data structure constant.
   vm->idle();

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
         str->append( c1 );
         ++pos;
      }

      str->append( c );
      ++pos;
      getOk = file->get( c );
   }

   vm->unidle();

   if ( ! getOk && ! file->eof() )
   {
      s_breakage( file );
   }

   // even if we read an empty line AND then we hit eof, we must return true,
   // so the program knows that the last line is empty.
   vm->regA().setBoolean( getOk || pos > 0 );
}

/*#
   @method grabLine Stream
   @brief Grabs a line of text encoded data.
   @optparam size Maximum count of characters to be read before to return anyway.
   @return A string containing the read line, or oob(0) when the file is over.
   @raise IoError on system errors.

   This function works as @a Stream.grabText, but if a new line is encountered,
   the read terminates. Returned string does not contain the EOL sequence.

   At EOF, the function returns an oob(0), which in normal tests is translated
   as "false", and that can be used to build sequences.

   @note An empty line is returned as an empty string. Please, notice that empty
   lines are returned as empty strings, and that empty strings, in Falcon, assume
   "false" value when logically evaluated. On loops where not checking for an
   explicit EOF to be hit in the stream, you will need to check for the returned
   value to be != 0, or not out of band.

   For example, a normal loop may look like:
   @code
   s = InputStream( "file.txt" )
   while (l = s.grabLine()) != 0
      > "LINE: ", l
   end	
   s.close()
   @endcode

   But it is possible to use the fuinction also in for/in loops:
   @code
   s = InputStream( "file.txt" )
   for line in s.grabLine: > "LINE: ", line
   s.close()
   @endcode

   Or even comprehensions:
   @code
   s = InputStream( "file.txt" )
   lines_in_file = List().comp( s.grabLine )
   s.close()
   @endcode

   @note As @a Stream.readLine recycles a pre-allocated buffer provided
   as parameter it is more efficient than grabLine, unless you need to
   store each line for further processing later on.
*/
FALCON_FUNC  Stream_grabLine ( ::Falcon::VMachine *vm )
{
   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );
   Item *i_size = vm->param(0);

   if ( i_size != 0 && ! i_size->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "[N]" ) );
   }


   int32 size = i_size == 0 ? -1 : (int32) i_size->forceInteger();
   if ( size == 0 )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->moduleString( rtl_zero_size ) ) );
   }

   // if the stream is at eof, we must return oob(0).
   if ( file->eof() )
   {
      vm->retval( (int64) 0 );
      vm->regA().setOob( true );
      return;
   }

   CoreString *str = new CoreString;
   // put it in the VM now, so that it can be inspected
   vm->retval( str );

   if ( size > 0 )
      str->reserve( size );

   vm->idle();
   uint32 c = 0, c1 = 0;
   int pos = 0;
   bool getOk = file->get( c );
   while( getOk && (size < 0 || pos < size) )
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
         str->append( c1 );
         ++pos;
      }

      str->append( c );
      ++pos;
      getOk = file->get( c );
   }

   vm->unidle();

   if ( ! getOk )
   {
      if ( ! file->eof() )
      {
         s_breakage( file );
      }
      // a last line with some data?
      else if( pos == 0 )
      {
         // no? -- consider it null
         vm->retval( (int64) 0 );
         vm->regA().setOob( true );
      }
   }
   // otherwise, let the returned string to go.

}

/*#
   @method write Stream
   @brief Write binary data to a stream.
   @param buffer A string or a MemBuf containing the data to be written.
   @optparam size Number of bytes to be written.
   @optparam start A position from where to start writing.
   @return Amount of data actually written.
   @raise IoError on system errors.

   Writes bytes from a buffer on the stream. The write operation is synchronous,
   and will block the VM until the stream has completed the write; however, the
   stream may complete only partially the operation. The number of bytes actually
   written on the stream is returned.

   When the output buffer is a string, a size parameter can be given; otherwise
   the whole binary contents of the stream are written. A start position may
   optionally be given too; this allows to iterate through writes and send part
   of the data that coulden't be send previously without extracting substrings or
   copying the memory buffers.

   MemBuf items can participate to stream binary writes through their internal
   position pointers. The buffer is written from @a MemoryBuffer.position up to
   @a MemoryBuffer.limit, and upon completion @a MemoryBuffer.position is advanced accordingly
   to the number of bytes effectively stored on the stream. When a MemBuf is
   used, @b size and @b start parameters are ignored.
*/

FALCON_FUNC  Stream_write ( ::Falcon::VMachine *vm )
{
   Item *i_source = vm->param(0);
   if ( i_source == 0 || ! ( i_source->isString() || i_source->isMemBuf() ))
   {
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "S|M, [N, N]" ) );
   }

   uint32 size, start;
   byte *buffer;

   if ( i_source->isString() )
   {
      Item *i_count = vm->param(1);
      Item *i_start = vm->param(2);
      if (
         ( i_count != 0 && ! i_count->isOrdinal() ) ||
         ( i_start != 0 && ! i_start->isOrdinal() )
      )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
            extra( "S|M, [N, N]" ) );
      }

      String* str = i_source->asString();

      if ( i_start == 0 )
      {
         start = 0;
      }
      else
      {
         start = (uint32) i_start->forceInteger();
         if ( start > str->size() )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ) );
         }
      }

      if( i_count == 0 )
      {
         size = str->size();
      }
      else
      {
         size = (uint32) i_count->forceInteger();
         if ( size > str->size() - start )
         {
            size = str->size() - start;
         }
      }

      buffer = str->getRawStorage();
   }
   else
   {
      MemBuf* mb = i_source->asMemBuf();
      start = mb->position();
      size = mb->limit() - start;
      buffer = mb->data();
   }

   Stream *file = dyncast<Stream *>( vm->self().asObject()->getFalconData() );

   vm->idle();
   int64 written = file->write( buffer + start, size );
   vm->unidle();

   if ( written < 0 )
   {
      s_breakage( file );
   }

   if ( i_source->isMemBuf() )
      i_source->asMemBuf()->position(  (uint32) (i_source->asMemBuf()->position() + written) );

   vm->retval( written );
}

/*#
   @method writeText Stream
   @brief Write text data to a stream.
   @param buffer A string containing the text to be written.
   @optparam start The character count from which to start writing data.
   @optparam end The position of the last character to write.
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
                     vm->self().asObject()->getFalconData() );

   Item *source = vm->param(0);
   Item *begin = vm->param(1);
   Item *end = vm->param(2);
   uint32 iBegin, iEnd;

   if ( source == 0 || source->type() != FLC_ITEM_STRING ||
      (begin != 0 && begin->type() != FLC_ITEM_INT ) ||
      (end != 0 && end->type() != FLC_ITEM_INT ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "S,[N],[N]" ) );
   }

   iBegin = begin == 0 ? 0 : (uint32) begin->asInteger();
   iEnd = end == 0 ? source->asString()->length() : (uint32) end->asInteger();

   vm->idle();
   if ( ! file->writeString( *(source->asString()), iBegin, iEnd )  )
   {
      vm->unidle();
      s_breakage( file );
   }
   else {
      vm->unidle();
   }
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
                     vm->self().asObject()->getFalconData() );

   Item *position = vm->param(0);
   if ( position== 0 || ! position->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   vm->idle();
   int64 pos = file->seekBegin( position->forceInteger() );
   vm->unidle();

   if ( file->bad() )
   {
      s_breakage( file );
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
   a negative number meaning "backward", and a positive meaning "forward". If the
   stream does not support seeking, an IoError is raised. If the operation would
   move the pointer past the end of the file size, the pointer is set to the end;
   if it would move the pointer before the beginning, it is moved to the beginning.
   On success, the function returns the position where the pointer has been moved.
*/
FALCON_FUNC  Stream_seekCur ( ::Falcon::VMachine *vm )
{
   Stream *file = static_cast<Stream *>(
                     vm->self().asObject()->getFalconData() );

   Item *position = vm->param(0);
   if ( position== 0 || ! position->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   vm->idle();
   int64 pos = file->seekCurrent( position->forceInteger() );
   vm->unidle();

   if ( file->bad() )
   {
      s_breakage( file );
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
                     vm->self().asObject()->getFalconData() );

   Item *position = vm->param(0);
   if ( position== 0 || ! position->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   vm->idle();
   int64 pos = file->seekEnd( position->forceInteger() );
   vm->unidle();

   if ( file->bad() )
   {
      s_breakage( file );
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
                     vm->self().asObject()->getFalconData() );

   vm->idle();
   int64 pos = file->tell();
   vm->unidle();

   if ( file->bad() )
   {
      s_breakage( file );
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
                     vm->self().asObject()->getFalconData() );

   Item *position = vm->param(0);
   int64 pos;

   // declaring the VM idle from now on.
   VMachine::Pauser pauser( vm );

   if ( position == 0 )
      pos = file->tell();
   else if ( ! position->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   else
      pos = position->forceInteger();

   if ( pos == -1 || ! file->truncate( pos ) )
   {
      s_breakage( file );
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
                     vm->self().asObject()->getFalconData() );
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
                     vm->self().asObject()->getFalconData() );
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
                     vm->self().asObject()->getFalconData() );
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
                     vm->self().asObject()->getFalconData() );
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
                     vm->self().asObject()->getFalconData() );

   Item *secs_item = vm->param(0);
   int32 msecs = secs_item == 0 ? -1 : (int32) (secs_item->forceNumeric()*1000);

   if ( msecs != 0 ) vm->idle();
   int32 avail = file->readAvailable( msecs, &vm->systemData() );
   if ( msecs != 0 ) vm->unidle();

   if ( file->interrupted() )
   {
      vm->interrupted( true, true, true );
      return;
   }

   if ( avail > 0 )
      vm->regA().setBoolean( true );
   else if ( avail == 0 )
      vm->regA().setBoolean( false );
   else if ( file->lastError() != 0 )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
         .origin( e_orig_runtime )
         .sysError( (uint32) file->lastError() ) );
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
                     vm->self().asObject()->getFalconData() );

   Item *secs_item = vm->param(0);
   int32 msecs = secs_item == 0 ? -1 : (int32) (secs_item->forceNumeric()*1000);

   if ( msecs != 0 ) vm->idle();
   int32 available = file->writeAvailable( msecs, &vm->systemData() );
   if ( msecs != 0 ) vm->unidle();

   if ( available <= 0 )
   {
      if ( file->interrupted() )
      {
         vm->interrupted( true, true, true );
         return;
      }

      if ( file->lastError() != 0 )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
            .origin( e_orig_runtime )
            .sysError( (uint32) file->lastError() ) );
      }
      vm->regA().setBoolean( false );
   }
   else {
      vm->regA().setBoolean( true );
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
                     vm->self().asObject()->getFalconData() );

   // create a new stream instance.
   Item *clstream = vm->findWKI( "Stream" );
   fassert( clstream != 0 );

   Stream *nstream = static_cast<Stream *>( file->clone() );
   // in case of filesystem error, we get 0 and system error properly set.
   if ( nstream == 0 )
   {
       throw new CodeError( ErrorParam( e_uncloneable )
            .origin( e_orig_runtime ) );
   }

   CoreObject *obj = clstream->asClass()->createInstance( nstream );
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
                     vm->self().asObject()->getFalconData() );
   CoreString *str = new CoreString;
   file->errorDescription( *str );
   vm->retval( str );
}


/*#
   @funset core_stream_factory Stream factory functions
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

   See @a core_stream_factory for a description of the shared modes.
*/
FALCON_FUNC  InputStream_creator ( ::Falcon::VMachine *vm )
{
   Item *fileName = vm->param(0);
   Item *fileShare = vm->param(1);

   if ( fileName == 0 || ! fileName->isString() ||
        ( fileShare != 0 && ! fileShare->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S,[N]" ) );
   }

   URI furi( *fileName->asString() );

   if ( !furi.isValid() )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *fileName->asString() )  );
   }

   // find the appropriage provider.
   VFSProvider* vfs = Engine::getVFS( furi.scheme() );
   if ( vfs == 0 )
   {
      throw new ParamError( ErrorParam( e_unknown_vfs, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *fileName->asString() )  );
   }

   ::Falcon::BaseFileStream::t_shareMode shMode = fileShare != 0 ?
         (::Falcon::BaseFileStream::t_shareMode) fileShare->forceInteger() :
         ::Falcon::BaseFileStream::e_smShareFull;

   VFSProvider::OParams params;
   params.rdOnly();

   if ( shMode == BaseFileStream::e_smExclusive )
      params.shNone();
   else if ( shMode == BaseFileStream::e_smShareRead )
      params.shNoWrite();

   vm->idle();
   Stream *in = vfs->open( furi, params );
   vm->unidle();

   if ( in == 0 )
   {
      throw vfs->getLastError();
   }

   Item *stream_class = vm->findWKI( "Stream" );
   //if we wrote the std module, can't be zero.
   fassert( stream_class != 0 );

   ::Falcon::CoreObject *co = stream_class->asClass()->createInstance( in );
   vm->retval( co );
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

   See @a core_stream_factory for a description of the shared modes.
*/
FALCON_FUNC  OutputStream_creator ( ::Falcon::VMachine *vm )
{
   Item *fileName = vm->param(0);
   if ( fileName == 0 || ! fileName->isString() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   URI furi( *fileName->asString() );

   if ( !furi.isValid() )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *fileName->asString() )  );
   }

   // find the appropriage provider.
   VFSProvider* vfs = Engine::getVFS( furi.scheme() );
   if ( vfs == 0 )
   {
      throw new ParamError( ErrorParam( e_unknown_vfs, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *fileName->asString() )  );
   }

   Item *osMode = vm->param(1);
   int mode;

   if ( osMode == 0 ) {
      mode = 0666;
   }
   else
   {
      if ( ! osMode->isOrdinal() ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      mode = (int) osMode->forceInteger();
   }

   Item *fileShare = vm->param(2);
   if ( fileShare != 0 && ! fileShare->isInteger() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   ::Falcon::BaseFileStream::t_shareMode shMode = fileShare != 0 ?
         (::Falcon::BaseFileStream::t_shareMode) fileShare->forceInteger() :
         ::Falcon::BaseFileStream::e_smShareFull;

   VFSProvider::CParams params;
   params.truncate();
   params.wrOnly();

   if ( shMode == BaseFileStream::e_smExclusive )
      params.shNone();
   else if ( shMode == BaseFileStream::e_smShareRead )
      params.shNoWrite();

   params.createMode( mode );

   vm->idle();
   Stream *stream =  vfs->create( furi, params );
   vm->unidle();

   if ( stream == 0 )
   {
      throw vfs->getLastError();
   }

   Item *stream_class = vm->findWKI( "Stream" );
   //if we wrote the std module, can't be zero.
   fassert( stream_class != 0 );
   CoreObject *co = stream_class->asClass()->createInstance(stream);
   vm->retval( co );
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

   See @a core_stream_factory for a description of the shared modes.
*/
FALCON_FUNC  IOStream_creator ( ::Falcon::VMachine *vm )
{
   Item *fileName = vm->param(0);
   if ( fileName == 0 || ! fileName->isString() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   URI furi( *fileName->asString() );

   if ( !furi.isValid() )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *fileName->asString() ) );
   }

   // find the appropriage provider.
   VFSProvider* vfs = Engine::getVFS( furi.scheme() );
   if ( vfs == 0 )
   {
      throw new ParamError( ErrorParam( e_unknown_vfs, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *fileName->asString() ) );
   }

   Item *osMode = vm->param(1);
   int mode;

   if ( osMode == 0 ) {
      mode = 0666;
   }
   else
   {
      if ( ! osMode->isOrdinal() ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      mode = (int) osMode->forceInteger();
   }

   Item *fileShare = vm->param(2);
   if ( fileShare != 0 && ! fileShare->isInteger() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   ::Falcon::BaseFileStream::t_shareMode shMode = fileShare != 0 ?
         (::Falcon::BaseFileStream::t_shareMode) fileShare->forceInteger() :
         ::Falcon::BaseFileStream::e_smShareFull;

   VFSProvider::CParams params;
   params.rdwr();

   if ( shMode == BaseFileStream::e_smExclusive )
      params.shNone();
   else if ( shMode == BaseFileStream::e_smShareRead )
      params.shNoWrite();

   params.createMode( mode );

   vm->idle();
   Stream *stream = vfs->open( furi, params );
   vm->unidle();

   if ( stream == 0 )
   {
      stream = vfs->create( furi, params );
      if ( stream == 0 )
         throw vfs->getLastError();
   }

   Item *stream_class = vm->findWKI( "Stream" );
   //if we wrote the std module, can't be zero.
   fassert( stream_class != 0 );
   ::Falcon::CoreObject *co = stream_class->asClass()->createInstance( stream );
   vm->retval( co );
}

static CoreObject *internal_make_stream( VMachine *vm, FalconData *clone, int userMode )
{
   // The clone stream may be zero if the embedding application doesn't want
   // to share a virtual standard stream with us.
   if ( clone == 0 )
   {
       throw new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) );
   }

   Item *stream_class;
   if ( userMode < 0 )
      stream_class = vm->findWKI( "Stream" );
   else
      stream_class = vm->findWKI( "StdStream" );

   //if we wrote the RTL module, can't be zero.
   fassert( stream_class != 0 );
   CoreObject *co = stream_class->asClass()->createInstance(clone);
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
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      //keep the stream
      internal_make_stream( vm, vm->stdIn()->clone(), -1 ); // this also returns the old stream

      Stream *orig = (Stream *) p1->asObject()->getFalconData();
      Stream *clone = (Stream *) orig->clone();
      if ( clone == 0 )
      {
         throw new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) );
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
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      //keep the stream
      internal_make_stream( vm, vm->stdOut()->clone(), -1 );
      Stream *orig = (Stream *) p1->asObject()->getFalconData();
      Stream *clone = (Stream *) orig->clone();
      if ( clone == 0 )
      {
         throw new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) );
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
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
      }

      //keep the stream
      internal_make_stream( vm, vm->stdErr()->clone(), -1 );
      Stream *orig = (Stream *) p1->asObject()->getFalconData();
      Stream *clone = (Stream *) orig->clone();
      if ( clone == 0 )
      {
         throw new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) );
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

   The stream is write only. Read operations will cause an IoError to be raised.
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

/*# @endset */

/*#
   @function systemErrorDescription
   @ingroup general_purpose
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   CoreString *str = new CoreString;
   ::Falcon::Sys::_describeError( number->forceInteger(), *str );
   vm->retval( str );
}



/*#
   @method setBuffering Stream
   @brief Set the buffering state of this stream.
   @param size Buffering size; pass 0 to disable.

   This method activates or disactivates I/O buffering on this stream.

   When buffering is active, every read/write operation is first cached
   in memory, provided the size of the memory buffer is wide enough to
   store the data being written or to provide the data being read.

   Seek operations invalidate the buffer, that is automatically flushed
   when necessary.

   Local filesystem providers and standard I/O streams are buffered
   by default; other streams may be created with buffering enabled or not,
   buffered or not depending on their and common usage patterns (network
   streams are usually unbuffered).

   You may want to disable buffering when preparing binary data in memory,
   or parsing big chunks of binary data at once via block (MemBuf) read/write
   operations. However, notice that buffering is always optimizing when chunk width is
   1/4 of the buffer size or less, and causes only minor overhead in the other cases.
*/

FALCON_FUNC  Stream_setBuffering ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Stream *file = dyncast<Stream *>( self->getFalconData() );

   Item *i_size = vm->param(0);

   if ( i_size == 0 || ! i_size->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N") );
   }

   int64 size = i_size->forceInteger();

   // Buffering lies between transcoders and the final stream.
   // We need to find the first non-transcoder in the hierarcy.

   Stream *sub = file;
   Transcoder* ts = 0;   // lowermost transcoder.
   while( sub->isTranscoder() )
   {
      ts = dyncast<Transcoder*>( sub );
      sub = ts->underlying();
   }

   // detach the lowermost transcoder.
   if ( ts != 0 )
      ts->detach();

   // Ok, the sub stream may be a buffering stream or the final stream itself.
   if ( sub->isStreamBuffer() )
   {
      StreamBuffer* sb = dyncast<StreamBuffer*>( sub );
      // do we want to disable it?
      if( size <= 0 )
      {
         sub = sb->underlying();
         sb->flush();
         sb->detach();
         delete sb;

         // re-attach the unbuffered stream to the transcoders,
         //    or set it as the new user data.
         if( ts != 0 )
         {
            // reflected streams are always owned.
            ts->setUnderlying( sub, true );
         }
         else
            self->setUserData( sub );
      }
      else
      {
         // we just want to resize it.
         if ( ! sb->resizeBuffer( (uint32) size ) )
         {
            throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .origin( e_orig_runtime )
               .sysError( (uint32) sb->underlying()->lastError() ) );
         }
      }
   }
   else
   {
      // if we want to enable it, we need to create a bufferer.
      if ( size > 0 )
      {
         StreamBuffer* sb = new StreamBuffer( sub, true, (uint32) size );
         // attach the newly buffered stream to the transcoders,
         //    or set it as the new user data.
         if( ts != 0 )
         {
            // reflected streams are always owned.
            ts->setUnderlying( sb, true );
         }
         else
            self->setUserData( sb );
      }
   }
}

/*#
   @method getBuffering Stream
   @brief Returns the size of I/O buffering active on this stream.
   @return 0 if the stream is unbuffered or a positive number if it is buffered.

   @see Stream.setBuffering
*/

FALCON_FUNC  Stream_getBuffering ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Stream *file = dyncast<Stream *>( self->getFalconData() );

   // Buffering lies between transcoders and the final stream.
   // We need to find the first non-transcoder in the hierarcy.
   Stream *sub = file;
   while( sub->isTranscoder() )
   {
      Transcoder* ts = dyncast<Transcoder*>( sub );
      sub = ts->underlying();
   }

   // if it's a streambuffer, we have buffering enabled
   if( sub->isStreamBuffer() )
   {
      StreamBuffer* sb = dyncast<StreamBuffer*>( sub );
      vm->retval( (int64) sb->bufferSize() );
   }
   else {
      // buffering is disabled.
      vm->retval( 0 );
   }
}

/*#
   @method setEncoding Stream
   @brief Set the text encoding and EOL mode for text-based operations.
   @param encoding Name of the encoding that is used for the stream.
   @optparam EOLMode How to treat end of line indicators.

   This method sets an encoding that will affect readText() and writeText() methods.
   Provided encodings are:
   - "utf-8"
   - "utf-16"
   - "utf-16BE"
   - "utf-16LE"
   - "iso8859-1" to "iso8859-15"
   - "gbk" (Chinese simplified)
   - "cp1252"
   - "C" (byte oriented -- writes byte per byte)

   As EOL manipulation is also part of the text operations, this function allows to
   chose how to deal with EOL characters stored in Falcon strings when writing data
   and how to parse incoming EOL. Available values are:
   - CR_TO_CR: CR and LF characters are untranslated
   - CR_TO_CRLF: When writing, CR ("\n") is translated into CRLF, when reading CRLF is translated into a single "\n"
   - SYSTEM_DETECT: use host system policy.

   If not provided, this parameter defaults to SYSTEM_DETECT.

   If the given encoding is unknown, a ParamError is raised.
*/

FALCON_FUNC  Stream_setEncoding ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Stream *file = dyncast<Stream *>( self->getFalconData() );

   Item *i_encoding = vm->param(0);
   Item *i_eolMode = vm->param(1);

   if ( i_encoding == 0 || ! i_encoding->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   int mode = ( i_eolMode == 0 ? SYSTEM_DETECT : (int) i_eolMode->forceInteger());
   if( mode != SYSTEM_DETECT && mode != CR_TO_CR && mode != CR_TO_CRLF )
   {
      mode = SYSTEM_DETECT;
   }

   // find the first non-transcoder stream entity
   while( file->isTranscoder() )
   {
      Transcoder* tc = dyncast<Transcoder *>(file);
      file = tc->underlying();
      tc->detach();
      delete tc;
   }

   // just in case, set the stream back in place.
   self->setUserData( file );

   Transcoder *trans = TranscoderFactory( *(i_encoding->asString()), file, true );

   if ( trans == 0 )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ) );
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

/*#
   @function readURI
   @brief Reads fully data from a given file or URI source.
   @param uri The item to be read (URI or string)
   @optparam encoding The encoding.
   @return A string containing the whole contents of the
          given file.
   @raise IoError in case of read error.

   This function reads as efficiently as possible a file
   from the given source. If encoding isn't given,
   the file is read as binary data.

   Provided encodings are:
   - "utf-8"
   - "utf-16"
   - "utf-16BE"
   - "utf-16LE"
   - "iso8859-1" to "iso8859-15"
   - "gbk" (Chinese simplified)
   - "cp1252"
   - "C" (byte oriented -- writes byte per byte)

   @note The maximum size of the data that can be read is
   limited to 2 Gigabytes.
*/
FALCON_FUNC  readURI ( ::Falcon::VMachine *vm )
{
#define READURI_READ_BLOCK_SIZE 2048

   Item *i_uri = vm->param(0);
   Item *i_encoding = vm->param(1);

   URI uri;

   if ( i_encoding != 0 && ! ( i_encoding->isString()|| i_encoding->isNil()) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "S|URI, [S]" ) );
   }

   if ( i_uri->isString() )
   {
      uri.parse( *i_uri->asString() );
      if( ! uri.isValid() ) {
         throw new ParamError( ErrorParam( e_malformed_uri, __LINE__ )
            .extra( *i_uri->asString() ) );
      }
   }
   else if ( i_uri->isOfClass( "URI" ) )
   {
      uri = dyncast<UriObject*>( i_uri->asObjectSafe() )->uri();
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "S|URI, [S]" ) );
   }


   // find the appropriage provider.
   VFSProvider* vfs = Engine::getVFS( uri.scheme() );
   if ( vfs == 0 )
   {
      throw new ParamError( ErrorParam( e_unknown_vfs, __LINE__ )
            .extra( uri.scheme() ) );
   }

   // Idling the VM here; we're starting to access the system.
   vm->idle();

   Stream *in = vfs->open( uri, VFSProvider::OParams().rdOnly() );
   if ( in == 0 )
   {
      vm->unidle();
      throw vfs->getLastError();
   }

   FileStat fs;
   int64 len;
   if( ! vfs->readStats( uri, fs ) )
   {
      // we know the file exists; this means that the vfs doesn't provide file lenght.
      len = -1;
   }
   else {
      len = fs.m_size;
   }

   String *ret = new CoreString();

   // direct read?
   if ( i_encoding == 0 || i_encoding->isNil() )
   {
      if ( len >= 0 && len < 2000000000 )
         ret->reserve( (uint32) len );

      int64 pos = 0;
      while( ! in->eof() )
      {
         int rin = 0;
         if ( len >= 0 && len < 2000000000 )
         {
            rin = in->read( ret->getRawStorage() + pos, (int32)(len+1) ); // so we hit immediately EOF
         }
         else {
            ret->reserve( (int32) pos + READURI_READ_BLOCK_SIZE );
            rin = in->read( ret->getRawStorage() + pos, READURI_READ_BLOCK_SIZE );
         }

         if ( rin < 0 )
         {
            vm->unidle();
            int64 fsError = in->lastError();
            delete in;

            throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .extra( uri.get() )
               .sysError( (int32) fsError ) );
         }

         pos += rin;
      }
      ret->size( (uint32) pos );
      delete in;
   }
   else {
      // text read.
      Stream* tin = TranscoderFactory( *i_encoding->asString(), in, true );
      String temp;

      while( ! tin->eof() )
      {
         bool res;
         if ( len >= 0 && len < 2000000000 )
         {
            res = tin->readString( temp, (uint32) len );
         }
         else {
            res = tin->readString( temp, READURI_READ_BLOCK_SIZE );
         }

         if ( ! res )
         {
            vm->unidle();
            int64 fsError = tin->lastError();
            delete in;

            throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .extra( uri.get() )
               .sysError( (uint32) fsError ) );
         }

         ret->append( temp );
      }

      delete tin;
   }

   vm->unidle();
   vm->retval( ret );
}

/*#
   @function writeURI
   @brief Writes fully data to a given file or URI source.
   @param uri The item to be read (URI or string)
   @param data A string or membuf containing the data to be written.
   @optparam encoding The encoding.
   @raise IoError in case of write errors.

   This function writes all the data contained in the string or
   memory buffer passed in the @b data parameter into the @b uri
   output resource.

   If @b encoding is not given, the data is treated as binary data
   and written as-is.

   Provided encodings are:
   - "utf-8"
   - "utf-16"
   - "utf-16BE"
   - "utf-16LE"
   - "iso8859-1" to "iso8859-15"
   - "gbk" (Chinese simplified)
   - "cp1252"
   - "C" (byte oriented -- writes byte per byte)

*/

FALCON_FUNC  writeURI ( ::Falcon::VMachine *vm )
{
#define WRITEURI_WRITE_BLOCK_SIZE 2048

   Item *i_uri = vm->param(0);
   Item *i_data = vm->param(1);
   Item *i_encoding = vm->param(2);

   URI uri;

   if ( i_data == 0 || ! ( i_data->isString() )
      || (i_encoding != 0 && ! (i_encoding->isString() || i_encoding->isNil())) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "S|URI, S, [S]" ) );
   }

   if ( i_uri->isString() )
   {
      uri.parse( *i_uri->asString() );
      if( ! uri.isValid() ) {
         throw new ParamError( ErrorParam( e_malformed_uri, __LINE__ )
            .extra( *i_uri->asString() ) );
      }
   }
   else if ( i_uri->isOfClass( "URI" ) )
   {
      uri = dyncast<UriObject*>( i_uri->asObjectSafe() )->uri();
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "S|URI, S, [S]" ) );
   }

   // find the appropriate provider.
   VFSProvider* vfs = Engine::getVFS( uri.scheme() );
   if ( vfs == 0 )
   {
      throw new ParamError( ErrorParam( e_unknown_vfs, __LINE__ )
            .extra( uri.scheme() ) );
   }

   // Idling the VM here; we're starting to access the system.
   vm->idle();

   VFSProvider::CParams params;
   params.truncate();
   params.wrOnly();

   Stream *out = vfs->create( uri, params );
   if ( out == 0 )
   {
      vm->unidle();
      throw vfs->getLastError();
   }

   // direct read?
   if ( i_encoding == 0 || i_encoding->isNil() )
   {
      byte* data = i_data->asString()->getRawStorage();
      uint32 size =  i_data->asString()->size();

      uint32 pos = 0;
      while( pos < size )
      {
         int wout = out->write( data + pos, size - pos ); // so we hit immediately EOF

         if ( wout < 0 )
         {
            vm->unidle();
            int64 fsError = out->lastError();
            delete out;

            throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .extra( uri.get() )
               .sysError( (uint32) fsError ) );
         }

         pos += wout;
      }

      delete out;
   }
   else {
      // text read.
      Stream* tout = TranscoderFactory( *i_encoding->asString(), out, true );
      if ( ! tout->writeString( *i_data->asString() ) )
      {
         vm->unidle();
         int64 fsError = tout->lastError();
         delete tout;

         throw new IoError( ErrorParam( e_io_error, __LINE__ )
            .extra( uri.get() )
            .sysError( (uint32) fsError ) );
      }
      delete tout;
   }

   vm->unidle();
}

}}
/* end of file.cpp */
