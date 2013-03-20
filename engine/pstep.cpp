/*
   FALCON - The Falcon Programming Language.
   FILE: pstep.cpp

   Common interface to VM processing step.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 18:01:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC 
#define SRC "engine/pstep.cpp"

#include <falcon/pstep.h>

namespace Falcon {

void PStep::describeTo( String& s ) const
{
   s = "Unnamed pstep";
}

String PStep::renderPrefix(int32 size)
{
   if( size > 0 )
   {
      String s(" ");
      return s.replicate(size * depthIndent);
   }
   else {
      return "";
   }
}

}

/* end of pstep.cpp */
