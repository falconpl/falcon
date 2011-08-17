/*
   FALCON - The Falcon Programming Language.
   FILE: modgroupbase.h

   Base abastract class for module groups and spaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MODGROUPBASE_H_
#define _FALCON_MODGROUPBASE_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/modmap.h>
#include <falcon/symbolmap.h>

namespace Falcon 
{

/** Base class for module group and module space.
 */
class FALCON_DYN_CLASS ModGroupBase
{
public:
   ModGroupBase();
   virtual ~ModGroupBase();
      
   const ModMap& modules() const { return m_modules; }
   ModMap& modules() { return m_modules; }
   
   const SymbolMap& symbols() const { return m_symbols; }
   SymbolMap& symbols() { return m_symbols; }
   
protected:   
   
   ModMap m_modules;
   SymbolMap m_symbols;
};

}

#endif	/* _FALCON_MODGROUPBASE_H_ */

/* end of modgroupbase.h */
