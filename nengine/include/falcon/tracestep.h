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
   uint32 m_pc;
   String m_modpath;

public:
   //TODO: Remove this version in the next version.
   TraceStep( const String &module, const String symbol, uint32 line, uint32 pc ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_pc( pc )
   {}

   TraceStep( const String &module, const String &mod_path, const String symbol, uint32 line, uint32 pc ):
      m_module( module ),
      m_symbol( symbol ),
      m_line( line ),
      m_pc( pc ),
      m_modpath( mod_path )
   {}

   const String &module() const { return m_module; }
   const String &modulePath() const { return m_modpath; }
   const String &symbol() const { return m_symbol; }
   uint32 line() const { return m_line; }
   uint32 pcounter() const { return m_pc; }

   String toString() const { String temp; return toString( temp ); }
   String &toString( String &target ) const;
};

}

#endif

/* end of tracestep.h */
