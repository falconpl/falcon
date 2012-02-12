/*
   FALCON - The Falcon Programming Language.
   FILE: path.h

   Falcon core module -- Interface to Path.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_RESTORER_H
#define FALCON_CORE_RESTORER_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>

#include <falcon/usercarrier.h>
#include <falcon/path.h>

namespace Falcon {
namespace Ext {

class ClassRestorer: public ClassUser
{
public:
   
   ClassRestorer();
   virtual ~ClassRestorer();
   
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void* createInstance() const;
   
   /*
   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;
    */
private:   
   
   FALCON_DECLARE_PROPERTY( hasNext );
   FALCON_DECLARE_METHOD( next, "" );
   FALCON_DECLARE_METHOD( restore, "stream:Stream" );
};

}
}

#endif

/* end of restorer.h */
