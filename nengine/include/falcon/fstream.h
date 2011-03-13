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
   FStream( const FStream &other );
   virtual ~FStream();

   virtual int32 read( void *buffer, int32 size );
   virtual int32 write( const void *buffer, int32 size );

   virtual int64 seek( int64 pos, Stream::e_whence whence );
   virtual int64 tell();
   virtual bool truncate(int64 pos = - 1);

   virtual int32 readAvailable( int32 msecs_timeout=0, Interrupt* intr=0 );
   virtual int32 writeAvailable( int32 msecs_timeout=0, Interrupt* intr=0 );

   virtual bool close();

   virtual FStream* clone() const;

protected:
   FStream( void *fsdata );

   void *m_fsData;
};

/** File stream with output functions filtered out. */
class FALCON_DYN_CLASS InputOnlyFStream: public FStream
{
public:
   InputOnlyFStream( void *fsdata ):
      FStream( fsdata )
   {}

   InputOnlyFStream( const InputOnlyFStream& other ):
      FStream( other )
   {}


   virtual ~InputOnlyFStream() {}

   virtual int32 writeAvailable( int32 msecs_timeout=0, Interrupt* intr=0 );
   virtual int32 write( const void *buffer, int32 size );
   virtual bool truncate(int64 pos = - 1);

   virtual InputOnlyFStream* clone() const;
};


/** File stream with input functions filtered out. */
class FALCON_DYN_CLASS OutputOnlyFStream: public FStream
{
public:
   OutputOnlyFStream( void *fsdata ):
      FStream( fsdata )
   {}

   OutputOnlyFStream( const OutputOnlyFStream& other ):
      FStream( other )
   {}

   virtual ~OutputOnlyFStream() {}

   virtual int32 readAvailable( int32 msecs_timeout=0, Interrupt* intr=0 );
   virtual int32 read( const void *buffer, int32 size );

   virtual OutputOnlyFStream* clone() const;
};


/** File stream with output and seek functions filtered out. */
class FALCON_DYN_CLASS ReadOnlyFStream: public InputOnlyFStream
{
public:
   ReadOnlyFStream( void *fsdata ):
      InputOnlyFStream( fsdata )
   {}

   ReadOnlyFStream( const ReadOnlyFStream& other ):
      InputOnlyFStream( other )
   {}

   virtual ~ReadOnlyFStream() {}

   virtual int64 seek( int64 pos, Stream::e_whence whence );
   virtual int64 tell();

   virtual ReadOnlyFStream* clone() const;
};


/** File stream with  input and seek functions filtered out.*/
class FALCON_DYN_CLASS WriteOnlyFStream: public OutputOnlyFStream
{
public:

   WriteOnlyFStream( void *fsdata ):
      OutputOnlyFStream( fsdata )
   {}

   WriteOnlyFStream( const WriteOnlyFStream& other ):
      OutputOnlyFStream( other )
   {}

   virtual ~WriteOnlyFStream() {}

   virtual int64 seek( int64 pos, Stream::e_whence whence );
   virtual int64 tell();
   virtual bool truncate(int64 pos = - 1);

   virtual WriteOnlyFStream* clone() const;
};

}

#endif

/* end of fstream.h */
