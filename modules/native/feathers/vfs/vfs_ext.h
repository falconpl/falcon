/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_ext.h

   Interface to Falcon Virtual File System -- various
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 21 Mar 2013 22:23:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_EXT_H
#define FALCON_FEATHERS_VFS_EXT_H

#include <falcon/setup.h>
#include <falcon/function.h>

/*#
 @beginmodule vfs
 */
namespace Falcon {
namespace Ext {


/*#
  @function InputStream
  @brief Opens an input stream
  @param uri The URI to be opened
  @return An open input straem
  @raise IOError on error.

  This function opens a logical input stream using an URI that is sent to the
  virtual file system resolver. If the protocol is not specified, "file://"
  (meaning: local file system) is assumed.

  For a finer control, use system-specific extension or the @a vfs module.

  @deprecated
*/

FALCON_DECLARE_FUNCTION(InputStream, "uri:S");


/*#
  @function OutputStream
  @brief Opens an output stream
  @param uri The URI to be opened
  @return An open input straem
  @raise IOError on error.

  This function opens a logical output stream using an URI that is sent to the
  virtual file system resolver. If the protocol is not specified, "file://"
  (meaning: local file system) is assumed.

  Generally, this resolves in creating a new file, or truncating to zero
  an existing one.

  For a finer control, use system-specific extension or the @a vfs module.

  @deprecated
*/

FALCON_DECLARE_FUNCTION(OutputStream, "uri:S");


/*#
  @function IOStream
  @brief Creates a stream opened for input and output
  @param uri The URI to be opened
  @return An open input straem
  @raise IOError on error.

  This function opens a logical input/output stream using an URI that is sent to the
  virtual file system resolver. If the protocol is not specified, "file://"
  (meaning: local file system) is assumed.

  Generally, this resolves in opening a file for append, if it exist, or creating
  a new file if it doesn't exist.

  For a finer control, use system-specific extension or the @a vfs module.

  @deprecated
*/

FALCON_DECLARE_FUNCTION(IOStream, "uri:S");

}
}


#endif

/* end of vfs_ext.h */

