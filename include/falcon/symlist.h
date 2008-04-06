/*
   FALCON - The Falcon Programming Language.
   FILE: symlist.h

   Declaration of a list of symbols.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 3 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Declaration of a list of symbols.
*/

#ifndef flc_symlist_H
#define flc_symlist_H

#include <falcon/genericlist.h>

namespace Falcon {

class FALCON_DYN_CLASS SymbolList: public List
{
public:
   SymbolList()
   {}

   SymbolList( const SymbolList &other );

};

}

#endif

/* end of symlist.h */
