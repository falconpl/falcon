/*
   FALCON - The Falcon Programming Language.
   FILE: directory.h

   Internal functions prototypes for DirApi.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:07:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Internal functions prototypes for DirApi.

   This files holds the internal api for directories that is not to be published.
*/

#ifndef _FALCON_DIRECTORY_H_
#define _FALCON_DIRECTORY_H_

#include <falcon/uri.h>

namespace Falcon {

/** Base abstract class for directory enumerator handlers.
 
 This class is a base interface for the virtual file systems providing
 the enumeration of file entries that can be accessed under a certain URI.

 The \b file VFS opens directories on the local hard disk.

 Implementations of this class are hidden from the final user and are
 provided directly by the virtual file system that allows to access a certain
 feature.

*/
class Directory
{
public:
   /** The base destructor calls close(). */
   virtual ~Directory();

   /** Reads the next directory entry.
    \param fname A String where to place the name of the next file in the directory.
    \return true if another directory can be read, false if this was the last entry.
    \throw I/O Error in case of read error.
    */
   virtual bool read( String &fname ) = 0;

   /** Closes the directory handle.
    Further read() requests after this call shall return false.
    */
   virtual void close() = 0;

   /** Returns the URI to which this directory handle refers. */
   const URI &path() const { return m_uri; }

protected:
   /** Creates a directory entry referring to a certain URI. */
   Directory( const URI& uri ):
      m_uri( uri )
   {}

   URI m_uri;
};

}

#endif

/* end of directory.h */
