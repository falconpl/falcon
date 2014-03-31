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

String &TraceStep::toString( String &target, bool bAddPath, bool bAddParams ) const
{
   target += m_module;
   if ( bAddPath && ! m_modpath.empty() )
   {
      target += "(" + m_modpath + ")";
   }

   if( m_line > 0 )
   {
      target += ":";
      target.writeNumber( (int64) m_line );
   }

   target += " ";
   target += m_symbol;

   if( bAddParams && ! m_rparams.empty() )
   {
      target += "(";
      target += m_rparams;
      target += ")";
   }

   return target;
}

}

/* endo f tracestep.cpp */
