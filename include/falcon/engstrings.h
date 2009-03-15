/*
   FALCON - The Falcon Programming Language.
   FILE: engstrings.h

   Declarations of engine string table
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar feb 13 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Declarations of engine string table
*/

#ifndef flc_engstrings_H
#define flc_engstrings_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/globals.h>
#include <falcon/string.h>

namespace Falcon {

class FALCON_DYN_SYM StringTable;

const String &getMessage( uint32 id );
bool setTable( StringTable *tab );

/** Sets the engine language.
   Searches the given language definition in ISO format (i.e. en_US or it_IT)
   in the string tables that the engine uses as message source.
   The string table may be built in or searched on the disk *TODO*.
*/
bool FALCON_DYN_SYM setEngineLanguage( const String &language );

}

#endif

/* end of engstrings.h */
