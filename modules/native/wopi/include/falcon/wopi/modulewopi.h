/*
   FALCON - The Falcon Programming Language.
   FILE: modulewopi.h

   Web Oriented Programming Interface main module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Oct 2013 14:34:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_WOPI_MODULEWOPI_H
#define FALCON_WOPI_MODULEWOPI_H

namespace Falcon {
namespace WOPI {

/**
   Specific WOPI interface Falcon Module
*/
class ModuleWopi: public Module
{
public:
   ModuleWopi(WOPI* w);
   virtual ~ModuleWopi();

   WOPI* wopi() const { return m_wopi; }
private:
   WOPI* m_wopi;
};

}
}

#endif

/* modulewopi.h */
