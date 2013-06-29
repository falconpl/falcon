/*
   FALCON - The Falcon Programming Language.
   FILE: debugger.h

   Falcon core module -- Interface to the collector.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 06 Mar 2013 17:40:44 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_DEBUGGER_H
#define FALCON_CORE_DEBUGGER_H

#include <falcon/class.h>

namespace Falcon {
namespace Ext {

/*#
 @object Debugger
 */
class ClassDebugger: public Class
{
public:

   ClassDebugger();
   virtual ~ClassDebugger();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
};

}
}

#endif

/* end of debugger.h */
