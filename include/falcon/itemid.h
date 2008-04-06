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


#define FLC_ITEM_NIL       0
#define FLC_ITEM_INT       1
#define FLC_ITEM_NUM       2
#define FLC_ITEM_RANGE     3
#define FLC_ITEM_BOOL      4

#define FLC_ITEM_FUNC      6
/* Some shallow callable items. */
#define FLC_ITEM_FBOM      7

/** Special VM item */
#define FLC_ITEM_ATTRIBUTE 8
/** Used to store pointers in temporary local items by
   two-step VM functions. */
#define FLC_ITEM_POINTER   9

/** From this point on, we have possibly deep items */
#define FLC_ITEM_FIRST_DEEP 10

#define FLC_ITEM_STRING    10
#define FLC_ITEM_ARRAY     11
#define FLC_ITEM_DICT      12
#define FLC_ITEM_OBJECT    13
#define FLC_ITEM_MEMBUF    14

#define FLC_ITEM_REFERENCE 15

#define FLC_ITEM_CLSMETHOD 20
#define FLC_ITEM_METHOD    22
#define FLC_ITEM_CLASS     24


#define FLC_ITEM_INVALID   99

#endif

/* end of flc_itemid.h */
