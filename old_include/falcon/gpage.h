/*
   FALCON - The Falcon Programming Language.
   FILE: flc_itempage.h

   Definition of the page that holds garbageable data pointers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Definition of the page that holds garbageable data pointers.
*/

#ifndef flc_flc_gpage_H
#define flc_flc_gpage_H

#include <falcon/spage.h>
#include <falcon/garbageable.h>

namespace Falcon {

/** Garbageable pointer page.
   \note This is old code to be removed.
   A segregated allocation page holding references to items that may get garbaged if not marked.
*/
class GarbagePage: public SegregatedPage< Garbageable * >
{
public:
   GarbagePage( GarbagePage *prev = 0, GarbagePage *next = 0 ):
      SegregatedPage< Garbageable * >( prev, next )
   {}
};

class GPageList: public LinkedList< GarbagePage > {};
}

#endif

/* end of flc_gpage.h */
