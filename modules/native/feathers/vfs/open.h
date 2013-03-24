/*
   FALCON - The Falcon Programming Language.
   FILE: open.h

   Interface to Falcon Virtual File System -- open() function decl
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_OPEN_H
#define FALCON_FEATHERS_VFS_OPEN_H

#include <falcon/setup.h>
#include <falcon/function.h>

namespace Falcon {

class VFSModule;

namespace Ext {

/*# Opens a VFS entity.
 @param uri The VFS uri (string or URI entity) to be opened.
 @optparam mode Open read mode.
 @return On success a new stream.
 @raise IoError on error.

 The open mode could be an (bitwise-or) combination of the following:
 - @b O_RD: Read only
 - @b O_WR: Write only
 - @b O_APPEND: Set the file pointer at end
 - @b O_TRUNC:  Truncate

 - @b SH_NR: Shared read
 - @b SH_NW: Shared write

  The real meaning of the settings depends on the final
  virtual file system driver.

  By default the stream is opened as O_RD.
 */
class Open: public Function
{
public:
   Open( VFSModule* mod );
   virtual ~Open();   
   virtual void invoke( Falcon::VMContext* ctx, int );
   
private:
   VFSModule* m_module;
};

}
}


#endif
