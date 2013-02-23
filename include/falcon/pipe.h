/*
   FALCON - The Falcon Programming Language.
   FILE: fstream.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_FSTREAM_H_
#define _FALCON_FSTREAM_H_

#include <falcon/stream.h>

namespace Falcon {

/** Base class for file stream system data.
 * This is an empty class that is inherited by the concrete
 * system-specific file stream data. It is used to provide
 * an abstract representation of the raw system file structures,
 * descriptors or pointers, and associated utility data.
 *
 * The concrete implementation is different depending
 * on the target final system (the final system implementation
 * just declares the class).
 */

class SysFStreamData;

/** Base class for File descriptor streams.

 This streams wraps I/O on system low-level file descriptors. They can be integer
 descriptor on POSIX system or HANDLE object on MS-Windows, or other kind of atomic,
 system-level data on other systems.

 This low level I/O resource descriptiors usually map to opened files, pipes,
 sockets, standard process streams and so on.

 The FdStream hierarcy is subclassed by subtraction. Resources tipically exposing
 only some of the standard behavior are represented by subclasses where the method
 accessing to those functionalities are masked out; those method throw an UnsupportedException
 if invoked.
 
*/

class FALCON_DYN_CLASS FStream: public Stream
{
public:
   FStream( SysFStreamData* data );
   FStream( const FStream &other );
   virtual ~FStream();

   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );

   bool setNonblocking( bool );
   bool isNonbloking() const;

   virtual off_t seek( off_t pos, Stream::e_whence whence );
   virtual off_t tell();
   virtual bool truncate(off_t pos = - 1);

   virtual size_t readAvailable( int32 msecs_timeout=0 );
   virtual size_t writeAvailable( int32 msecs_timeout=0 );

   virtual bool close();

   virtual FStream* clone() const;

protected:
   SysFStreamData *m_fsData;
};

/** File stream with output functions filtered out. */
class FALCON_DYN_CLASS InputOnlyFStream: public FStream
{
public:
   InputOnlyFStream( SysFStreamData *fsdata ):
      FStream( fsdata )
   {}

   InputOnlyFStream( const InputOnlyFStream& other ):
      FStream( other )
   {}


   virtual ~InputOnlyFStream() {}

   virtual size_t writeAvailable( int32 msecs_timeout=0 );
   virtual size_t write( const void *buffer, size_t size );
   virtual bool truncate(int64 pos = - 1);

   virtual InputOnlyFStream* clone() const;
};


/** File stream with input functions filtered out. */
class FALCON_DYN_CLASS OutputOnlyFStream: public FStream
{
public:
   OutputOnlyFStream( SysFStreamData *fsdata ):
      FStream( fsdata )
   {}

   OutputOnlyFStream( const OutputOnlyFStream& other ):
      FStream( other )
   {}

   virtual ~OutputOnlyFStream() {}

   virtual size_t readAvailable( int32 msecs_timeout=0 );
   virtual size_t read( void *buffer, size_t size );

   virtual OutputOnlyFStream* clone() const;
};


/** File stream with output and seek functions filtered out. */
class FALCON_DYN_CLASS ReadOnlyFStream: public InputOnlyFStream
{
public:
   ReadOnlyFStream( SysFStreamData *fsdata ):
      InputOnlyFStream( fsdata )
   {}

   ReadOnlyFStream( const ReadOnlyFStream& other ):
      InputOnlyFStream( other )
   {}

   virtual ~ReadOnlyFStream() {}

   virtual off_t seek( off_t pos, Stream::e_whence whence );
   virtual off_t tell();

   virtual ReadOnlyFStream* clone() const;
};


/** File stream with  input and seek functions filtered out.*/
class FALCON_DYN_CLASS WriteOnlyFStream: public OutputOnlyFStream
{
public:

   WriteOnlyFStream( SysFStreamData *fsdata ):
      OutputOnlyFStream( fsdata )
   {}

   WriteOnlyFStream( const WriteOnlyFStream& other ):
      OutputOnlyFStream( other )
   {}

   virtual ~WriteOnlyFStream() {}

   virtual off_t seek( off_t pos, Stream::e_whence whence );
   virtual off_t tell();
   virtual bool truncate(off_t pos = - 1);

   virtual WriteOnlyFStream* clone() const;
};

}

#endif

/* end of fstream.h */
