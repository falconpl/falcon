/*
   FALCON - The Falcon Programming Language.
   FILE: classdirectory.h

   Falcon core module -- Handler to read VFS directory entries.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Mar 2013 00:25:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_CLASSDIRECTORY_H
#define FALCON_FEATHERS_VFS_CLASSDIRECTORY_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon {
namespace Ext {


class FALCON_DYN_CLASS ClassDirectory: public Class
{
public:
   ClassDirectory();
   virtual ~ClassDirectory();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};

}


}

#endif

/* end of classdirectory.h */
