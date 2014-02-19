/*
   FALCON - The Falcon Programming Language.
   FILE: log.h

   Falcon core module -- Log
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 21 Aug 2013 14:15:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_LOG_H
#define FALCON_CORE_LOG_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/item.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS ClassLog: public Class
{
public:
   
   ClassLog();
   virtual ~ClassLog();
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void* createInstance() const;   
};

}
}

#endif

/* end of log.h */
