/*
   FALCON - The Falcon Programming Language.
   FILE: wopi_ext.h

   Falcon Web Oriented Programming Interface.

   Main module generator
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 16:19:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_EXT_H
#define FALCON_WOPI_EXT_H

#include <falcon/module.h>
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon{
namespace WOPI {

#define FALCON_WOPI_SCRIPTNAME_PROP "scriptName"
#define FALCON_WOPI_SCRIPTPATH_PROP "scriptPath"

class ClassWopi: public Class
{
public:
   ClassWopi();
   virtual ~ClassWopi();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
};

}
}

#endif

/* end of wopi_ext.h */
