/*
   FALCON - The Falcon Programming Language.
   FILE: flc_itemid.h

   List of item ids
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 15 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   List if item ids.
*/

#ifndef flc_flc_itemid_H
#define flc_flc_itemid_H


#define FLC_ITEM_NIL          0
#define FLC_ITEM_BOOL         1
#define FLC_ITEM_INT          2
#define FLC_ITEM_NUM          3

/** From this point on, we have possibly deep items */
#define FLC_ITEM_FIRST_DEEP   4
#define FLC_ITEM_RANGE        4

#define FLC_ITEM_LBIND        5
#define FLC_ITEM_FUNC         6
/* Some shallow callable items. */

/** Used to store pointers in temporary local items by
   two-step VM functions. */
#define FLC_ITEM_GCPTR        7


#define FLC_ITEM_STRING       8
#define FLC_ITEM_ARRAY        9
#define FLC_ITEM_DICT         10
#define FLC_ITEM_OBJECT       11
#define FLC_ITEM_MEMBUF       12

#define FLC_ITEM_REFERENCE    13


#define FLC_ITEM_CLSMETHOD    14
#define FLC_ITEM_METHOD       15
#define FLC_ITEM_CLASS        16
#define FLC_ITEM_UNB          17

#define FLC_ITEM_COUNT        18
#define FLC_ITEM_INVALID      99

#endif

/* end of flc_itemid.h */
