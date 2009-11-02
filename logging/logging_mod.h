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


#ifndef flc_logging_mod_H
#define flc_logging_mod_H

#include <falcon/setup.h>
#include <falcon/error_base.h>
#include <falcon/corecarrier.h>
#include <falcon/srv/logging_srv.h>

#ifndef FALCON_LOGGING_ERROR_BASE
   #define FALCON_LOGGING_ERROR_BASE         1200
#endif

#define FALCON_LOGGING_ERROR_OPEN  (FALCON_LOGGING_ERROR_BASE + 0)

namespace Falcon
{

class LogChannelFilesCarrier: public CoreCarrier<LogChannelFiles>
{
public:
   LogChannelFilesCarrier( const CoreClass* gen, LogChannelFiles *data );
   LogChannelFilesCarrier( const LogChannelFilesCarrier& other );
   virtual ~LogChannelFilesCarrier();

   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &key, Item &ret ) const;
   virtual LogChannelFilesCarrier *clone() const;

};

CoreObject* LogChannelFilesFactory( const CoreClass *cls, void *data, bool );

}

#endif

/* end of logging_mod.h */
