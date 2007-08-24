/*
   FALCON - The Falcon Programming Language.
   FILE: flc_refstring.h
   $Id: refstring.h,v 1.1.1.1 2006/10/08 15:05:29 gian Exp $

   Referenced string
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer giu 9 2004
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
