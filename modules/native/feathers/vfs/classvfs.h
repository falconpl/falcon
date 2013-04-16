/*
   FALCON - The Falcon Programming Language.
   FILE: classvfs.h

   Falcon core module -- Interface to abstract virtual file system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Mar 2013 00:25:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_CLASSVFS_H
#define FALCON_FEATHERS_VFS_CLASSVFS_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/class.h>

#include <falcon/errors/paramerror.h>

namespace Falcon {
namespace Ext {

/*#
 @class VFS
 @brief Interface to abstract virtual file system.

 */
class ClassVFS: public Class
{
public:
   ClassVFS();
   virtual ~ClassVFS();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   static URI* internal_get_uri( Item* i_uri, URI& tempURI, Module* mod );
};

}


}

#endif

/* end of classvfs.h */
