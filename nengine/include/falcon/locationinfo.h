/*
   FALCON - The Falcon Programming Language.
   FILE: locationinfo.h

   Debug information about code location in a source file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_LOCATIONINFO_H
#define	_FALCON_LOCATIONINFO_H

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {


/** Debug information about code location in a source file.
 
 As it's inpractical to keep complete information about a source file
 in every VM step, it is necessary to reconstruct them when needed.

 This class holds the information that is needed by a third party debugger
 to pinpoint the current execution location in a source file.
*/
class FALCON_DYN_CLASS LocationInfo
{
public:
   String m_moduleName;
   String m_moduleUri;
   String m_function;
   int32 m_line;
   int32 m_char;
   
};

}

#endif

/* end of locationinfo.h */
