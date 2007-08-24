/*
   FALCON - The Falcon Programming Language.
   FILE: flc_itempage.h
   $Id: itempage.h,v 1.2 2007/01/10 00:37:52 jonnymind Exp $

   Definition of the page that holds items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 4 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Definition of the page that holds items.
*/

#ifndef flc_flc_itempage_H
#define flc_flc_itempage_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/item.h>

#define PAGE_SIZE 4096
#define FREE_PAGE_SPACE (PAGE_SIZE - sizeof( ItemPage *) - sizeof( ItemPage *) )
#define PAGE_ITEM_COUNT  (FREE_PAGE_SPACE/sizeof( Item ) )

namespace Falcon {

class MemPool;

/** Item page.
   A segregated allocation page holding Falcon VM items.
*/
class ItemPage: public BaseAlloc
{
public:
   ItemPage *m_prev;
   ItemPage *m_next;
   Item items[ PAGE_ITEM_COUNT ];
   // padding here
};

}

#endif

/* end of flc_itempage.h */
