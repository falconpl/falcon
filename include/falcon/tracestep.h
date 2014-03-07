/*
   FALCON - The Falcon Programming Language
   FILE: tracestep.h

   Error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 21:15:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_TRACESTEP_H
#define FALCON_TRACESTEP_H

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {

class FALCON_DYN_CLASS TraceStep
{
   String m_module;
   String m_symbol;
   uint32 m_line;
   String m_modpath;
   String m_rparams;

public:
   //TODO: Remove this version in the next version.
   TraceStep( const String &module, const String symbol, uint32 line, const String& rparams="" ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_rparams( rparams )
   {}

   TraceStep( const String &module, const String &mod_path, const String symbol, uint32 line, const String& rparams="" ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_modpath( mod_path ),
      m_rparams( rparams )
   {}

   const String &module() const { return m_module; }
   const String &modulePath() const { return m_modpath; }
   const String &symbol() const { return m_symbol; }
   uint32 line() const { return m_line; }
   const String& rparams() const { return m_rparams; }

   String toString() const { String temp; return toString( temp ); }
   String &toString( String &target ) const;
};

}

#endif

/* end of tracestep.h */
