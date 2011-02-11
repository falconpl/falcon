/*
   FALCON - The Falcon Programming Language.
   FILE: module.cpp

   Falcon code unit
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 14:37:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>

namespace Falcon {

Module::Module( const String& name ):
      m_name( name )
{}


Module::Module( const String& name, const String& uri ):
      m_name( name ),
      m_uri(uri)
{}

}

/* end of module.cpp */

