/*
   FALCON - The Falcon Programming Language.
   FILE: modmap.cpp

   A simple class orderly guarding modules -- private part.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_MODMAP_PRIVATE_H_
#define _FALCON_MODMAP_PRIVATE_H_

#include <falcon/modmap.h>
#include <falcon/string.h>
#include <falcon/mt.h>

#include <map>

namespace Falcon
{

class ModMap::Private
{
public:   
   typedef std::map< Module*, ModMap::Entry* > ModEntryMap;
   typedef std::map< String, ModMap::Entry* > NameEntryMap;
   
   ModEntryMap m_modMap;
   NameEntryMap m_modsByName;
   NameEntryMap m_modsByPath;   
   
   Mutex m_mtx;
   
   Private() {};
   
   ~Private()
   {
      // for all the modules
      ModEntryMap::iterator iter = m_modMap.begin();
      while( iter != m_modMap.end() )
      {
         // if we own the module, it will be delete as well
         delete iter->second; 
         ++iter;
      }
   }
};

}

#endif	/* _FALCON_MODMAP_PRIVATE_H_ */

/* end of modmap_private.h */
