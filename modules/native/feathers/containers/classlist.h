/* FALCON - The Falcon Programming Language.
 * FILE: classlist.h
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_FEATHERS_CLASSLIST_H_
#define _FALCON_FEATHERS_CLASSLIST_H_

#include "classcontainer.h"

namespace Falcon {

class ClassList: public ClassContainerBase
{
public:
   ClassList( const Class* cls );
   virtual ~ClassList();

   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};

}

#endif

/* end of containers_mod.h */

