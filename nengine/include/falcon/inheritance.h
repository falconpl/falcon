/*
   FALCON - The Falcon Programming Language.
   FILE: inheritance.h

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 13:35:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_INHERITANCE_H_
#define _FALCON_INHERITANCE_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

/** Structure holding information about inheritance in a class.

 */
class FALCON_DYN_CLASS Inheritance
{
public:
   ~Inheritance();
   
   //TODO
   const String& className() const { return m_name; }

private:
   String m_name;
};

}

#endif /* _FALCON_INHERITANCE_H_ */

/* end of inheritance.h */
