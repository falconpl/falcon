/*
   FALCON - The Falcon Programming Language.
   FILE: fbom.h
   $Id: fbom.h,v 1.3 2007/07/05 07:30:04 jonnymind Exp $

   Falcon basic object model
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer lug 4 2007
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
   Falcon basic object model.
   This file contains definition and helpers for the Falcon Basic Object Model,
   that is, the basic Object Oriented interface for items (and objects).

   FBOM is actually implemented as static tables of very simple items.

*/

#ifndef flc_fbom_H
#define flc_fbom_H

#include <falcon/setup.h>

namespace Falcon {

class Item;
class VMachine;

namespace Fbom
{

void toString( VMachine *vm, Item *elem, Item *format );
void makeIterator( VMachine *vm, const Item &self, bool begin );

}

}

#endif

/* end of fbom.h */
