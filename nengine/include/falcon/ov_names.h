/*
   FALCON - The Falcon Programming Language.
   FILE: ov_names.h

   Names of the methods that are used to override operators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Jul 2011 12:49:17 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_OV_NAMES_H
#define	_FALCON_OV_NAMES_H

#define OVERRIDE_OP_NEG       "__neg"

#define OVERRIDE_OP_ADD       "__add"
#define OVERRIDE_OP_SUB       "__sub"
#define OVERRIDE_OP_MUL       "__mul"
#define OVERRIDE_OP_DIV       "__div"
#define OVERRIDE_OP_MOD       "__mod"
#define OVERRIDE_OP_POW       "__pow"

#define OVERRIDE_OP_AADD      "__aadd"
#define OVERRIDE_OP_ASUB      "__asub"
#define OVERRIDE_OP_AMUL      "__amul"
#define OVERRIDE_OP_ADIV      "__adiv"
#define OVERRIDE_OP_AMOD      "__amod"
#define OVERRIDE_OP_APOW      "__apow"

#define OVERRIDE_OP_INC       "__inc"
#define OVERRIDE_OP_DEC       "__dec"
#define OVERRIDE_OP_INCPOST   "__incpost"
#define OVERRIDE_OP_DECPOST   "__decpost"

#define OVERRIDE_OP_CALL      "__call"

#define OVERRIDE_OP_GETINDEX  "__getIndex"
#define OVERRIDE_OP_SETINDEX  "__setIndex"
#define OVERRIDE_OP_GETPROP   "__getProperty"
#define OVERRIDE_OP_SETPROP   "__setProperty"

#define OVERRIDE_OP_COMPARE   "compare"
#define OVERRIDE_OP_ISTRUE    "__isTrue"
#define OVERRIDE_OP_IN        "__in"
#define OVERRIDE_OP_PROVIDES  "__provides"
#define OVERRIDE_OP_TOSTRING  "toString"
#define OVERRIDE_OP_FIRST     "__first"
#define OVERRIDE_OP_NEXT      "__next"

#define OVERRIDE_OP_COUNT 29



#define OVERRIDE_OP_NEG_ID       0

#define OVERRIDE_OP_ADD_ID       1
#define OVERRIDE_OP_SUB_ID       2
#define OVERRIDE_OP_MUL_ID       3
#define OVERRIDE_OP_DIV_ID       4
#define OVERRIDE_OP_MOD_ID       5
#define OVERRIDE_OP_POW_ID       6

#define OVERRIDE_OP_AADD_ID      7
#define OVERRIDE_OP_ASUB_ID      8
#define OVERRIDE_OP_AMUL_ID      9
#define OVERRIDE_OP_ADIV_ID      10
#define OVERRIDE_OP_AMOD_ID      11
#define OVERRIDE_OP_APOW_ID      12

#define OVERRIDE_OP_INC_ID       13
#define OVERRIDE_OP_DEC_ID       14
#define OVERRIDE_OP_INCPOST_ID   15
#define OVERRIDE_OP_DECPOST_ID   16

#define OVERRIDE_OP_CALL_ID      17

#define OVERRIDE_OP_GETINDEX_ID  18
#define OVERRIDE_OP_SETINDEX_ID  19
#define OVERRIDE_OP_GETPROP_ID   20
#define OVERRIDE_OP_SETPROP_ID   21

#define OVERRIDE_OP_COMPARE_ID   22
#define OVERRIDE_OP_ISTRUE_ID    23
#define OVERRIDE_OP_IN_ID        24
#define OVERRIDE_OP_PROVIDES_ID  25
#define OVERRIDE_OP_TOSTRING_ID  26
#define OVERRIDE_OP_FIRST_ID     27
#define OVERRIDE_OP_NEXT_ID      28

#define OVERRIDE_OP_COUNT_ID  29


#if OVERRIDE_OP_COUNT_ID != OVERRIDE_OP_COUNT
#error "Count of overrides was not the same of override name count"
#endif

#endif	/* _FALCON_OV_NAMES_H */

/* end of ov_names.h */
