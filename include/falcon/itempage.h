/*
   FALCON - The Falcon Programming Language.
   FILE: itempage.h

   A recyclable table of items that can be filled progressively
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 10 May 2013 16:23:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ITEMPAGE_H
#define FALCON_ITEMPAGE_H

#include <falcon/poolable.h>
#include <falcon/item.h>

namespace Falcon {

class ItemStack;

/** Recyclable table of items that can be filled progressively.

 This class represents the smallest unit of a growable page-based stack
 of items that are dynamically allocated in blocks of page.

 This is how it's supposed to be used:

 @code
   [RootItemPage]-->[ItemPage]-->[ItemPage]
 |          ^         ^ ^           ^
 |  {Item*}-|         | |           |
 |                    | |           |
 |  {Item*}-----------| |           |
 |  {Item*}-------------|           |
 |                                  |
 |  {Item*}-------------------------|
 V

@endcode

A rootItemPage is initially allocated.

To push a new item in the virtual stack, the getItem() method
is called; if it's not found, a new page is allocated from the pool

As items are popped from the virtual downward stack, possibly in block
(i.e. as popping a call frame stack removes many items at once), pages
left fully unused are returned to their pool.

Although this class can be used alone, the ItemStack class uses it
to provide a consistent interface.
*/
class FALCON_DYN_CLASS ItemPage: public Poolable
{
public:
   static const length_t ITEMS_IN_PAGE=32;

   inline ItemPage(length_t baseDepth = 0) {reset(baseDepth);}

   inline Item* getItem() {
      if (m_taken < ITEMS_IN_PAGE) return &m_items[m_taken++];
      return 0;
   }

   inline long depth() const { return m_baseDepth; }
   inline void depth( long v ) { m_baseDepth = v; }

   void reset( length_t baseDepth = 0 ) {
      m_taken = 0;
      m_baseDepth = baseDepth;
      m_prevPage = m_nextPage = 0;
   }

   void linkAfterPage( ItemPage* page )
   {
      page->m_nextPage = this;
      this->m_prevPage = page;
   }

protected:
   virtual ~ItemPage() {}
   
private:
   
   Item m_items[ITEMS_IN_PAGE];
   length_t m_taken;
   length_t m_baseDepth;
   ItemPage *m_prevPage;
   ItemPage *m_nextPage;

   friend class ItemStack;
};
}

#endif	/* FALCON_POOLABLE_H */

/* end of itempage.h */
