/*
   FALCON - The Falcon Programming Language.
   FILE: selector.h

   Falcon core module -- Semaphore shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 22:34:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_SELECTOR_H
#define FALCON_CORE_SELECTOR_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

namespace Falcon {
namespace Ext {

/*#
 @class Selector
 @brief Selects ready streams.
 @param fair True to create an acquirable (fair) selector.

 */
class FALCON_DYN_CLASS ClassSelector: public ClassShared
{
public:
   ClassSelector();
   virtual ~ClassSelector();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

};

}


}

#endif	

/* end of semaphore.h */
