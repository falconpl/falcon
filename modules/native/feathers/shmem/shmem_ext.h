/*
   FALCON - The Falcon Programming Language.
   FILE: shmem_ext.h

   Compiler module version informations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Nov 2013 13:11:01 +0100


   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FEATHERS_SHMEM_EXT_H_
#define _FALCON_FEATHERS_SHMEM_EXT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {

class ClassSharedMem: public Class
{
public:
   ClassSharedMem();
   virtual ~ClassSharedMem();

   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual int64 occupiedMemory( void* instance ) const;
};

}

#endif /* _FALCON_FEATHERS_SHMEM_EXT_H_ */

/* end of shmem_ext.h */
