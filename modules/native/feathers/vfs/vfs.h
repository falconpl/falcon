/*
   FALCON - The Falcon Programming Language.
   FILE: vfs.h

   Interface to Falcon Virtual File System -- main header
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_H
#define FALCON_FEATHERS_VFS_H

#include <falcon/module.h>

#define FALCON_VFS_MODE_FLAG_RAW 0x100

namespace Falcon {

class VFSModule: public Module
{
public:
   VFSModule();
   virtual ~VFSModule();
   
   static Falcon::Error* onURIResolved( Falcon::Module*, Falcon::Module* , Falcon::Symbol* );
   static Falcon::Error* onStreamResolved( Falcon::Module*, Falcon::Module* , Falcon::Symbol* );
  
   Class* uriClass() const { return m_uriClass; }
   Class* streamClass() const { return m_streamClass; }
   
private:
   Class* m_uriClass;
   Class* m_streamClass;
};

}

#endif

/* end of vfs.h */
