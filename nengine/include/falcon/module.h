/*
   FALCON - The Falcon Programming Language.
   FILE: module.h

   Falcon code unit
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 14:37:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MODULE_H
#define	FALCON_MODULE_H

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {

class FALCON_DYN_CLASS Module {
public:
   Module( const String& name );
   Module( const String& name, const String& uri );

   const String& name() const { return m_name; }
   const String& uri() const {return m_uri;}

public:
   String m_name;
   String m_uri;

};

}

#endif	/* MODULE_H */

/* end of module.h */
