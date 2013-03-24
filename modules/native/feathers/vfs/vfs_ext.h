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

namespace Falcon {

class VFSModule;

namespace Ext {

/*# Opens a read-only input stream.
 @param uri The VFS uri (string) to be opened.
 @return On success a new stream.
 @raise IoError on error.
@deprecated
 */
FALCON_DECLARE_FUNCTION(InputStream, "uri:S");

/*# Creates a new write-only output stream
 @param uri The VFS uri (string) to be opened.
 @return On success a new stream.
 @raise IoError on error.
@deprecated
 */
FALCON_DECLARE_FUNCTION(OutputStream, "uri:S");

/*# Opens a VFS entity.
 @param uri The VFS uri (string) to be opened.
 @return On success a new stream.
 @raise IoError on error.
 @deprecated
 */
FALCON_DECLARE_FUNCTION(IOStream, "uri:S");

}
}


#endif

/* end of vfs_ext.h */

