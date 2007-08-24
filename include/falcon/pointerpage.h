/*
   FALCON - The Falcon Programming Language.
   FILE: flc_itempage.h
   $Id: pointerpage.h,v 1.1.1.1 2006/10/08 15:05:38 gian Exp $

   Definition of the page that holds garbageable data pointers.
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
