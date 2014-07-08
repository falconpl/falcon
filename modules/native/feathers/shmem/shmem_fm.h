/*
   FALCON - The Falcon Programming Language.
   FILE: shmem_fm.h

   Compiler module version informations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Nov 2013 13:11:01 +0100


   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_FEATHERS_SHMEM_FM_H_
#define _FALCON_FEATHERS_SHMEM_FM_H_

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

class ModuleShmem: public Module
{
public:

   ModuleShmem();
   virtual ~ModuleShmem();

   Class* sessionClass() const { return m_classSession; }

   /** Reimplement to provide the "Session" service.
    *
    */
   virtual Service* createService( const String& name );

private:
   Class* m_classSession;
};

}
}

#endif /* _FALCON_FEATHERS_SHMEM_FM_H_ */

/* end of shmem_fm.h */
