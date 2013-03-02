/*
   FALCON - The Falcon Programming Language.
   FILE: stdstreams.h

   System default I/O streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Mar 2011 20:06:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_STDSTREAMS_H_
#define _FALCON_STDSTREAMS_H_

#include <falcon/fstream.h>

namespace Falcon {


/** Standard Input Stream proxy.
   This proxy opens a dupped stream that interacts with the standard stream of the process.
   The application (and the VM, and the scripts too) may open and close an arbitrary number of
   instances of this class.

 \note A single instance of this class may take ownership of the process standard stream passing the
 constructor parameter as false.

*/
class FALCON_DYN_CLASS StdInStream: public ReadOnlyFStream
{
public:
   StdInStream( bool bDup = true );
   virtual ~StdInStream() {}
   StreamTraits* traits() const;
};

/** Standard Output Stream proxy.
   This proxy opens a dupped stream that interacts with the standard stream of the process.
   The application (and the VM, and the scripts too) may open and close an arbitrary number of
   instances of this class.


    \note A single instance of this class may take ownership of the process standard stream passing the
 constructor parameter as false.
*/
class FALCON_DYN_CLASS StdOutStream: public WriteOnlyFStream
{
public:
   /** Creates a stream writing to the standard error stream of the host process.

    If bDup is set to false, close() will actually close the process stream.

    \param bDup pass as false to take the ownership of the underlying stream.
    */
   StdOutStream( bool bDup = true);
   virtual ~StdOutStream() {}
   StreamTraits* traits() const;
};

/** Standard Error Stream proxy.
   This proxy opens a dupped stream that interacts with the standard stream of the process.
   The application (and the VM, and the scripts too) may open and close an arbitrary number of
   this instances, without interfering each other.

   If a script, the VM or an embedding application (that wishes to do it through Falcon portable
   xplatform API) needs to close the standard stream, then it must create and delete (or simply close)
   an instance of RawStdxxxStream.
*/
class FALCON_DYN_CLASS StdErrStream: public WriteOnlyFStream
{
public:
   /** Creates a stream writing to the standard error stream of the host process.

    If bDup is set to false, close() will actually close the process stream.

    \param bDup pass as false to take the ownership of the underlying stream.
    */
   StdErrStream( bool bDup = true );
   virtual ~StdErrStream() {}
   StreamTraits* traits() const;
};

}

#endif

/* end of stdstreams.h */
