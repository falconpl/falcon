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

/** \file
   Definition for file based streams.

   This file contains both the FileStream definitions and the
   Standard Stream embeddings.

   The implementation is fully system dependant, so the common library will
   contain different implementation modules for different systems; in example:
      - fstream_sys_win.cpp - windows implementation
      - fstream_sys_unix.cpp - unixlike implementation
      ecc...
*/

#ifndef flc_fstream_H
#define flc_fstream_H

#include <falcon/stream.h>

namespace Falcon {

/**
   Class storing system specific file data.

   This class is used by generic file stream to store system
   dependent data.

   The implementors may use this class to open a file or a file-like
   handle with their own native preferred methods, and then store the
   handle inside the system-specific subclass of this class. The
   instance will then be passed to the appropriate Stream constructor.
*/
class FALCON_DYN_CLASS FileSysData: public BaseAlloc
{
protected:
   FileSysData() {}
public:
   virtual ~FileSysData() {}
   virtual FileSysData *dup() = 0;
};

/** File stream base class.
   The BaseFileStream class is the base for the system-specific stream handlers.
   It provides an interface to the system functions that work with
   system streams, as files or standard streams.
*/

class FALCON_DYN_CLASS BaseFileStream: public Stream
{
public:
   /** Open mode. */
    typedef enum {
      e_omReadWrite,
      e_omReadOnly,
      e_omWriteOnly
   } t_openMode;

   /** Share mode. */
   typedef enum  {
      e_smExclusive,
      e_smShareRead,
      e_smShareFull
   } t_shareMode;

   /** System attributes. */
   typedef enum {
      e_aOtherRead = 04,
      e_aOtherWrite = 02,
      e_aOtherExecute = 01,
      e_aGroupRead = 040,
      e_aGroupWrite = 020,
      e_aGroupExecute = 010,
      e_aUserRead = 0400,
      e_aUserWrite = 0200,
      e_aUserExecute = 0100,

      e_aAll = 0777,
      e_aReadOnly = 0444,
      e_aHidden = 0600
   } t_attributes;

protected:

   FileSysData *m_fsData;

   virtual int64 seek( int64 pos, Stream::e_whence whence );

public:
   BaseFileStream( t_streamType streamType, FileSysData *fsdata ):
      Stream( streamType ),
      m_fsData( fsdata )
   {}

   BaseFileStream( const BaseFileStream &other );

   virtual ~BaseFileStream();

   virtual bool close();
   virtual int32 read( void *buffer, int32 size );
   virtual int32 write( const void *buffer, int32 size );
   virtual int64 tell();
   virtual bool truncate( int64 pos = - 1);
   virtual bool errorDescription( ::Falcon::String &description ) const;
   const FileSysData *getFileSysData() const { return m_fsData; }
   virtual int64 lastError() const;
   virtual bool put( uint32 chr );
   virtual bool get( uint32 &chr );
   /**
      Return 1 if read available, 0 if not available, -1 on error;
   */
   virtual int32 readAvailable( int32 msecs_timeout, const Sys::SystemData *sysData = 0 );

   virtual int32 writeAvailable( int32 msecs_timeout, const Sys::SystemData *sysData = 0 );

   virtual bool writeString( const String &source, uint32 begin = 0, uint32 end = csh::npos );
   virtual bool readString( String &content, uint32 size );

   /** Set the error.
      Subclasses may use this to set error across platforms.
      The status is not changed.
   */
   void setError( int64 errorCode );

   virtual BaseFileStream *clone() const;
};

class FALCON_DYN_CLASS FileStream: public BaseFileStream
{
public:
   /** Constructs a stream based on system resources.
      The implementor must create a FileSysData suitable for the host system
      containing the resoruces the target system need to access files.
   */
   FileStream( FileSysData *fsdata ):
      BaseFileStream( t_file, fsdata )
   {
      status( t_open );
   }

   /** Default constructor.
      System specific implementations will create a consistent FileSysData
      representing an unopened system resource.

      Use this constructor if this stream must be used to manage a file that
      must be created or opened.

      If the stream is created or opened independently, use the
      FileStream( FileSysData *) constructor.
   */
   FileStream();

   virtual void setSystemData( const FileSysData &data );

   /** Open the file.
      On success, the internal file system specific data are filled with the newly
      created data. On failure, lastError() will return the error code.
   */
   virtual bool open( const String &filename, t_openMode mode=e_omReadOnly, t_shareMode share=e_smExclusive );

   /** Create the file.
      If the file already existed, it is destroyed and overwritten.

      On success, the internal file system specific data are filled with the newly
      created data. On failure, lastError() will return the error code.
   */
   virtual bool create( const String &filename, t_attributes mode,  t_shareMode share=e_smExclusive);

};

class FALCON_DYN_CLASS StdStream: public BaseFileStream
{
   virtual int64 seek( int64 pos, Stream::e_whence whence ) {
      m_status = t_unsupported;
      return -1;
   }

public:
   StdStream( FileSysData *fsdata ):
      BaseFileStream( t_stream, fsdata )
   {
      status( t_open );
   }

   /** The StdStream destructor.
      Overrides basic stream to avoid closing of the underlying stream.
      The stream must be explicitly closed.
   */
   virtual ~StdStream()
   {}

/*   virtual int64 tell() {
      m_status = t_unsupported;
      return -1;
   }
*/
   virtual bool truncate( int64 pos=-1 ) {
      m_status = t_unsupported;
      return false;
   }
};

class FALCON_DYN_CLASS InputStream: public StdStream
{
public:

   InputStream( FileSysData *fsdata ):
      StdStream( fsdata )
   {}

   virtual int32 write( const void *buffer, int32 size ) {
      m_status = t_unsupported;
      return -1;
   }

};

class FALCON_DYN_CLASS OutputStream: public StdStream
{
public:

   OutputStream( FileSysData *fsdata ):
      StdStream( fsdata )
   {}

   virtual int32 read( void *buffer, int32 size ) {
      m_status = t_unsupported;
      return -1;
   }
};

/** Standard Input Stream proxy.
   This proxy opens a dupped stream that interacts with the standard stream of the process.
   The application (and the VM, and the scripts too) may open and close an arbitrary number of
   this instances, without interfering each other.

   If a script, the VM or an embedding application (that wishes to do it through Falcon portable
   xplatform API) needs to close the standard stream, then it must create and delete (or simply close)
   an instance of RawStdxxxStream.
*/
class FALCON_DYN_CLASS StdInStream: public InputStream
{
public:
   StdInStream();
};

/** Standard Output Stream proxy.
   This proxy opens a dupped stream that interacts with the standard stream of the process.
   The application (and the VM, and the scripts too) may open and close an arbitrary number of
   this instances, without interfering each other.

   If a script, the VM or an embedding application (that wishes to do it through Falcon portable
   xplatform API) needs to close the standard stream, then it must create and delete (or simply close)
   an instance of RawStdxxxStream.
*/
class FALCON_DYN_CLASS StdOutStream: public OutputStream
{
public:
   StdOutStream();
};

/** Standard Error Stream proxy.
   This proxy opens a dupped stream that interacts with the standard stream of the process.
   The application (and the VM, and the scripts too) may open and close an arbitrary number of
   this instances, without interfering each other.

   If a script, the VM or an embedding application (that wishes to do it through Falcon portable
   xplatform API) needs to close the standard stream, then it must create and delete (or simply close)
   an instance of RawStdxxxStream.
*/
class FALCON_DYN_CLASS StdErrStream: public OutputStream
{
public:
   StdErrStream();
};

/** Standard Input Stream encapsulation.
   This Falcon Stream class encapsulates in a multiplatform and script wise class the real
   physical unerlying process standard stream.

   Whatever happens to an instance of this class, it will happen also to the embedding process
   stream. In example, a script willing to close the output stream to signal that there's no
   more data to be sent before its termiantion, may get an instance of the raw output class
   through the RTL function stdOutRaw() and then close it with the close() method.

   If the embedding application wishes to stop VM and scripts from accessing the real process
   standard stream, it may simply disable the stdInRaw() stdOutRaw() and stdErrRaw() functions
   by removing them from the RTL module before linking it in the VM.
*/
class FALCON_DYN_CLASS RawStdInStream: public InputStream
{
public:
   RawStdInStream();
};

/** Standard Output Stream encapsulation.
   This Falcon Stream class encapsulates in a multiplatform and script wise class the real
   physical unerlying process standard stream.

   Whatever happens to an instance of this class, it will happen also to the embedding process
   stream. In example, a script willing to close the output stream to signal that there's no
   more data to be sent before its termiantion, may get an instance of the raw output class
   through the RTL function stdOutRaw() and then close it with the close() method.

   If the embedding application wishes to stop VM and scripts from accessing the real process
   standard stream, it may simply disable the stdInRaw() stdOutRaw() and stdErrRaw() functions
   by removing them from the RTL module before linking it in the VM.
*/
class FALCON_DYN_CLASS RawStdOutStream: public OutputStream
{
public:
   RawStdOutStream();
};

/** Standard Error Stream encapsulation.
   This Falcon Stream class encapsulates in a multiplatform and script wise class the real
   physical unerlying process standard stream.

   Whatever happens to an instance of this class, it will happen also to the embedding process
   stream. In example, a script willing to close the output stream to signal that there's no
   more data to be sent before its termiantion, may get an instance of the raw output class
   through the RTL function stdOutRaw() and then close it with the close() method.

   If the embedding application wishes to stop VM and scripts from accessing the real process
   standard stream, it may simply disable the stdInRaw() stdOutRaw() and stdErrRaw() functions
   by removing them from the RTL module before linking it in the VM.
*/
class FALCON_DYN_CLASS RawStdErrStream: public OutputStream
{
public:
   RawStdErrStream();
};

inline BaseFileStream::t_attributes operator|(  BaseFileStream::t_attributes one, BaseFileStream::t_attributes two)
{
   return (BaseFileStream::t_attributes) ( ((uint32)one) | ((uint32)two) );
}

inline BaseFileStream::t_attributes operator&(  BaseFileStream::t_attributes one, BaseFileStream::t_attributes two)
{
   return (BaseFileStream::t_attributes) ( ((uint32)one) & ((uint32)two) );
}

inline BaseFileStream::t_attributes operator^(  BaseFileStream::t_attributes one, BaseFileStream::t_attributes two)
{
   return (BaseFileStream::t_attributes) ( ((uint32)one) & ((uint32)two) );
}

}

#endif

/* end of fstream.h */
