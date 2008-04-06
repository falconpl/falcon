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

#ifndef flc_flc_itempage_H
#define flc_flc_itempage_H

#include <falcon/spage.h>
#include <falcon/garbageable.h>

namespace Falcon {

/** Item page.
   A segregated allocation page holding Falcon VM items.
*/
class GarbagePage: public SegregatedPage< Garbageable * > {};

}

#endif

/* end of flc_itempage.h */
