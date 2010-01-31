/*
   FALCON - The Falcon Programming Language.
   FILE: utils.h

   Utilities for Falcon packager
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Jan 2010 12:42:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/stream.h>
#include <falcon/string.h>

#include <vector>

namespace Falcon
{
extern Stream* stdOut;
extern Stream* stdErr;

extern void (*message)( const String &msg );

void error( const String &msg );
void warning( const String &msg );
void setVerbose( bool mode );

void splitPaths( const String& path, std::vector<String>& tgt );
bool copyFile( const String& source, const String& dest );


}

/* end of utils.h */
