/*
   FALCON - The Falcon Programming Language.
   FILE: symbolmap_private.h

   A simple class orderly guarding symbols and the module they come from.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYMBOLMAP_PRIVATE_H_
#define _FALCON_SYMBOLMAP_PRIVATE_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/symbolmap.h>

#include <map>

namespace Falcon
{

class SymbolMap::Private
{
public:
   typedef std::map<String, SymbolMap::Entry*> SymModMap;
   SymModMap m_syms;

   Mutex m_mtx;
   
   Private() {}
   ~Private() 
   {
      SymModMap::iterator iter = m_syms.begin();
      while( iter != m_syms.end() )
      {
         delete iter->second;
         ++iter;
      }
   }
};

}

#endif	/* _FALCON_SYMBOLMAP_PRIVATE_H_ */

/* end of symbolmap_private.h */
