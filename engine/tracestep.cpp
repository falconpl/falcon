/*
   FALCON - The Falcon Programming Language
   FILE: tracestep.cpp

   Error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 21:15:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/tracestep.h>

namespace Falcon {

String &TraceStep::toString( String &target ) const
{
   if ( m_modpath.size() )
   {
      target += "\"" + m_modpath + "\" ";
   }

   target += m_module + "." + m_symbol + ":";
   target.writeNumber( (int64) m_line );
   return target;
}

}

/* endo f tracestep.cpp */