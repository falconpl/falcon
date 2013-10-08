/*
   FALCON - The Falcon Programming Language.
   FILE: fwdata.h

   Falcon Web Oriented Programming Interface

   Framework data manager and framework data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 19 Apr 2010 20:51:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_FWDATA_H_
#define FALCON_WOPI_FWDATA_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/mt.h>

#include <map>

namespace Falcon {
namespace WOPI {

//============================================================
// Global framework data.
//

class FWData: public CoreObject
{
public:
      FWData();
};

class FWDataManager
{
public:
   FWDataManager();
   FWDataManager( const String& persistLocation );

   
private:
   Mutex m_mtx;

   typedef std::map< String, FWData > DataMap;
   DataMap m_mData;
};

}
}

/* end of fwdata.h */
