/*
   FALCON - The Falcon Programming Language.
   FILE: itemstack.h

   A growable stack of non-relocable items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 10 May 2013 16:23:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_ITEMSTACK_H_
#define _FALCON_ITEMSTACK_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>

namespace Falcon {

class ItemPage;

/* A growable stack of non-relocable items.


 */
class FALCON_DYN_CLASS ItemStack
{
public:
   ItemStack( Pool* pool );
   virtual ~ItemStack();

   ItemPage* freeUpToDepth( length_t depth );

   Item* push( length_t depth = 0 );
   Item* push( length_t depth, const Item& toBeCopied ) {
      Item* ret = push(depth);
      *ret = toBeCopied;
      return ret;
   }

   void pop();
   void pop(length_t count);

private:
   ItemPage* m_head;
   ItemPage* m_tail;
   Pool* m_pool;
};

}

#endif /* _FALCON_ITEMSTACK_H_ */

/* end of itemstack.h */

