/*
   FALCON - The Falcon Programming Language.
   FILE: options.h

   Options for Falcon packager
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Jan 2010 12:42:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

#include <set>

namespace Falcon
{

class Options
{
public:
   bool m_bPackFam;
   bool m_bStripSources;
   bool m_bNoSysFile;
   String m_sRunner;
   String m_sTargetDir;
   String m_sLoadPath;
   String m_sMainScript;
   String m_sEncoding;
   String m_sMainScriptPath;
   String m_sSystemRoot;
   String m_sFalconBinDir;
   String m_sFalconLibDir;
   bool m_bHelp;
   bool m_bVersion;
   bool m_bVerbose;

   Options();

   bool parse( int argc, char* const argv[] );
   bool isValid() const { return m_bIsValid; }
   bool isBlackListed( const String& modname ) const;
   bool isSysModule( const String& modname ) const;

private:
   bool m_bIsValid;
   std::set<String> m_blackList;
   std::set<String> m_sysModules;
};

}

/* end of options.h */
