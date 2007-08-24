/*
   FALCON - The Falcon Programming Language.
   FILE: itm_deref.h
   $Id: itm_deref.h,v 1.1.1.1 2006/10/08 15:05:40 gian Exp $

   Inline implementation of Item::dereference( byte code )
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu Feb 3 2005
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
    Inline implementation of Item::dereference( byte code )

    As the pcodes may vary greatly and this inline function is needed only in a
    small set of internal files (i.e. vm_run.cpp), the declaration of this inline
    function has been moved to a separate file.
*/

#ifndef flc_itm_deref_H
#define flc_itm_deref_H

#include <falcon/item.h>
#include <falcon/pcodes.h>

namespace Falcon {

inline Item *Item::dereference( byte opcode ) const {
   if ( type() != FLC_ITEM_REFERENCE ) //|| opcode >= P_TRAV )
      return this;
   return &asReference()->origin();
}

}

#endif

/* end of itm_deref.h */
