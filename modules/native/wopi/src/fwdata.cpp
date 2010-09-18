/*
   FALCON - The Falcon Programming Language.
   FILE: fwdata.cpp

   Falcon Web Oriented Programming Interface

   Framework data manager and framework data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 19 Apr 2010 20:51:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/fwdata.h>

#define FALCON_SESSION_DATA_EXT ".fsd"


namespace Falcon {
namespace WOPI {

//============================================================
// File based session data
//

FWDataManager::FWDataManager( const String& SID, const String& tmpDir ):
   SessionData( SID ),
   m_tmpDir( tmpDir )
{
}

}
}

/* end of fwdata.cpp */
