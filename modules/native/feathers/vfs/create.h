/*
   FALCON - The Falcon Programming Language.
   FILE: create.h

   Interface to Falcon Virtual File System -- create() function decl
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_CREATE_H
#define FALCON_FEATHERS_VFS_CREATE_H

#include <falcon/setup.h>
#include <falcon/function.h>

namespace Falcon {

class VFSModule;

namespace Ext {

/*# Opens a VFS entity.
 @param uri The VFS uri (string or URI entity) to be opened.
 @optparam mode Open read mode.

 */
class Create: public Function
{
public:
   Create( VFSModule* mod );
   virtual ~Create();   
   virtual void invoke( Falcon::VMContext* ctx, int );
   
private:
   VFSModule* m_module;
};

}
}


#endif	
