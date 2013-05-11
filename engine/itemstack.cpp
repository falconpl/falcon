/*
   FALCON - The Falcon Programming Language.
   FILE: itemstack.cpp

   A growable stack of non-relocable items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 10 May 2013 16:23:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/itemstack.cpp"

#include <falcon/itemstack.h>
#include <falcon/itempage.h>
#include <falcon/engine.h>

#include <falcon/errors/codeerror.h>

namespace Falcon {

ItemStack::ItemStack( Pool* pool ):
         m_pool(pool)
{
   m_head = m_tail = static_cast<ItemPage*>(pool->get());
   if( m_head == 0 )
   {
      m_head = m_tail = new ItemPage();
      m_head->assignToPool(m_pool);
   }
}

ItemStack::~ItemStack()
{
   ItemPage* page = m_head;
   while( page != 0 )
   {
      ItemPage* next = page->m_nextPage;
      page->dispose();
      page = next;
   }
}


ItemPage* ItemStack::freeUpToDepth( length_t depth )
{
   ItemPage *page = m_tail;
   while( page != 0 && page->m_baseDepth > depth )
   {
      ItemPage* prev = page->m_prevPage;
      page->dispose();
      page = prev;
   }

   m_tail = page;
   page->m_nextPage = 0;
   return page;
}


Item* ItemStack::push(length_t depth)
{
   if( m_tail->m_taken == ItemPage::ITEMS_IN_PAGE )
   {
      ItemPage* page = static_cast<ItemPage*>(m_pool->get());
      if( page == 0 ) {
         page = new ItemPage(depth);
         page->assignToPool(m_pool);
      }
      else {
         page->reset();
      }

      page->linkAfterPage(m_tail);
      m_tail = page;
   }

   return &m_tail->m_items[m_tail->m_taken++];
}


void ItemStack::pop()
{
   if( m_tail->m_taken == 0 )
   {
      if( m_tail == m_head )
      {
         throw FALCON_SIGN_XERROR( CodeError, e_stackuf, .extra("ItemStack::pop"));
      }
      else {
         m_tail = m_tail->m_prevPage;
         m_tail->m_nextPage->dispose();
         m_tail->m_nextPage = 0;
      }
   }
   else {
      m_tail->m_taken--;
   }
}

void ItemStack::pop( length_t count )
{
   while( count > m_tail->m_taken && m_tail != m_head )
   {
      count -= m_tail->m_taken;
      m_tail = m_tail->m_prevPage;
      m_tail->m_nextPage->dispose();
      m_tail->m_nextPage = 0;
   }


   if( count > m_tail->m_taken )
   {
      throw FALCON_SIGN_XERROR( CodeError, e_stackuf, .extra("ItemStack::pop(count)"));
   }

   m_tail->m_taken -= count;
}

}

/* end of itemstack.cpp */

