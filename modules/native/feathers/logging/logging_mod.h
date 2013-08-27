/*
   FALCON - The Falcon Programming Language.
   FILE: logging_mod.h

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_LOGGING_MOD_H
#define FALCON_FEATHERS_LOGGING_MOD_H

#include <falcon/setup.h>
#include <falcon/error_base.h>
#include <falcon/module.h>

#ifndef FALCON_LOGGING_ERROR_BASE
   #define FALCON_LOGGING_ERROR_BASE         1200
#endif

#define FALCON_LOGGING_ERROR_OPEN  (FALCON_LOGGING_ERROR_BASE + 0)
#define FALCON_LOGGING_ERROR_DESC  "Error opening the logging service"

namespace Falcon {

namespace Mod {
   class LogArea;
}

namespace Ext {

class LoggingModule: public Module
{
public:
   LoggingModule();
   virtual ~LoggingModule();

   Class* classLogArea()  const { return m_logArea; }
   Class* classLogChannel()  const { return m_logChannel; }
   Mod::LogArea* genericArea() const { return m_generalArea; }
private:

   Class* m_logArea;
   Class* m_logChannel;
   Mod::LogArea* m_generalArea;
};

}
}

#endif

/* end of logging_mod.h */
