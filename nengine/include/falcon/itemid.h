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

/* First, the flat items */

#define FLC_ITEM_NIL          0
#define FLC_ITEM_BOOL         1
#define FLC_ITEM_INT          2
#define FLC_ITEM_NUM          3
#define FLC_ITEM_FUNC         4
#define FLC_ITEM_METHOD       5
#define FLC_ITEM_BASEMETHOD   6
#define FLC_ITEM_COUNT        7

/** User items are non-standard items. */
#define FLC_ITEM_USER         8
/** Framing items are special markers in data stack for rules. */
#define FLC_ITEM_FRAMING      9


// Theese are the class IDs, used to get the typeID of this item.
#define FLC_CLASS_ID_STRING   10
#define FLC_CLASS_ID_FUNCTION 11
#define FLC_CLASS_ID_ARRAY    12
#define FLC_CLASS_ID_DICT     13
#define FLC_CLASS_ID_RANGE    14
#define FLC_CLASS_ID_CLASS    15
#define FLC_CLASS_ID_PROTO    16

// Other generic object
#define FLC_CLASS_ID_OBJECT   20


#define FLC_ITEM_INVALID      99

#endif

/* end of itemid.h */
