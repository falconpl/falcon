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

#ifndef FALCON_CORE_STORER_H
#define FALCON_CORE_STORER_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>

#include <falcon/usercarrier.h>
#include <falcon/path.h>

namespace Falcon {
namespace Ext {

class ClassStorer: public ClassUser
{
public:
   
   ClassStorer();
   virtual ~ClassStorer();
   
   void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void* createInstance( Item* params, int pcount ) const;   
private:   
   
   FALCON_DECLARE_METHOD( store, "item:S" );
   FALCON_DECLARE_METHOD( commit, "stream:Stream" );
};

}
}

#endif

/* end of storer.h */
