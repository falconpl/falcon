/*
   FALCON - The Falcon Programming Language.
   FILE: dynunloader.h

   Native dynamic library unloader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DYNUNLOADER_H_
#define _FALCON_DYNUNLOADER_H_

#include <falcon/setup.h>

namespace Falcon
{

class Module;
class String;

/** Native dynamic library unloader.
 
 This class is used by the engine to unload (discharge) Falcon modules
 loaded from native shared libraries.
 
 This class wraps a system-specific dynamic library handle, and provides
 an interface to the system routines to free the resources allocated through
 that handle.
 
 This class is created by the DynLoader and stored in the generated Module.
 At module destruction, right before freeing the memory associated with the 
 Falcon module, will invoke the unload() method of this class and discharge
 the data from the memory.
 
 This class is implmented in dynloader.cpp and related system-specific
 extensions.
 
 */
class FALCON_DYN_CLASS DynUnloader
{
public:
   DynUnloader( void* sysData );
   ~DynUnloader();     
   void unload();

private:

   void* m_sysData;
};

}

#endif	/* _FALCON_DYNUNLOADER_H_ */

/* end of dynunloader.h */
