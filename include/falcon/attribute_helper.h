/*
   FALCON - The Falcon Programming Language.
   FILE: attribute_helper.h

   Little helper for attribute declaration.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Jan 2013 21:00:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ATTRIBUTE_HELPER_H
#define FALCON_ATTRIBUTE_HELPER_H

namespace Falcon {
class VMContext;
class String;
class Mantra;
class TreeStep;

bool attribute_helper(VMContext* vmctx, const String& name, TreeStep* generator, Mantra* target );

}

#endif

/* end of attribute_helper.h */
