/*
   FALCON - The Falcon Programming Language.
   FILE: flc_refstring.h

   Referenced string
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer giu 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This file defines the StringRef type.
*/

#ifndef FALCON_STRINGREF_H
#define FALCON_STRINGREF_H

#include <falcon/hstring.h>
#include <falcon/refcount.h>

namespace Falcon
{

typedef Refcounter< Hstring > StringRef;

}

#endif
/* end of flc_refstring.h */
