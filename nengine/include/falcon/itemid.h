/*
   FALCON - The Falcon Programming Language.
   FILE: itemid.h

   List of item ids
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 15 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ITEMID_H
#define FALCON_ITEMID_H


#define FLC_ITEM_NIL          0
#define FLC_ITEM_BOOL         1
#define FLC_ITEM_INT          2
#define FLC_ITEM_NUM          3

#define FLC_ITEM_FLAT         3

/*
 Theese are hybrid flat-deep items.
 They are flat with regards to the item system, that is, they are
 copied flat, while they are deep with regards to the GC, that is,
 they require ad-hoc marking.
*/
#define FLC_ITEM_FUNC         4
#define FLC_ITEM_METHOD       5
#define FLC_ITEM_BASEMETHOD   6

/** From this point on, we have deep items */
#define FLC_ITEM_DEEP         7

#define FLC_ITEM_COUNT        8
#define FLC_ITEM_INVALID      99

#endif

/* end of itemid.h */
