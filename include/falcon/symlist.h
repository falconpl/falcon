/*
   FALCON - The Falcon Programming Language.
   FILE: symlist.h
   $Id: symlist.h,v 1.4 2007/04/02 20:28:08 jonnymind Exp $

   Declaration of a list of symbols.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 3 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
