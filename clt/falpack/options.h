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

namespace Falcon
{

class Options
{
public:
   bool m_bPackFam;
   bool m_bStripSources;
   bool m_bNoSysFile;
   bool m_bUseFalrun;
   String m_sTargetDir;
   String m_sLoadPath;
   String m_sMainScript;
   String m_sEncoding;
   String m_sMainScriptPath;
   bool m_bHelp;
   bool m_bVersion;
   bool m_bVerbose;

   Options();

   bool parse( int argc, char* const argv[] );
   bool isValid() const { return m_bIsValid; }

private:
   bool m_bIsValid;
};

}

/* end of options.h */
