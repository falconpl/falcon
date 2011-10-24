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


#endif	/* OPEN_H */
