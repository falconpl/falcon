/*
   FALCON - The Falcon Programming Language.
   FILE: shmemmodule.h

   Compiler module version informations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Nov 2013 13:11:01 +0100


   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_FEATHERS_SHMEM_MODULE_EXT_H_
#define _FALCON_FEATHERS_SHMEM_MODULE_EXT_H_

#include <falcon/module.h>

namespace Falcon {

class ShmemModule: public Module
{
public:

   ShmemModule();
   virtual ~ShmemModule();

   Class* sessionClass() const { return m_classSession; }

   /** Reimplement to provide the "Session" service.
    *
    */
   virtual Service* createService( const String& name );

private:
   Class* m_classSession;
};

}


#endif /* SHMEMMODULE_H_ */

/* end of shmemmodule.h */
