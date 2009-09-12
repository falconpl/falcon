/*
   FALCON - The Falcon Programming Language
   FILE: logging_mod.cpp

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "logging_mod.h"

namespace Falcon
{

LogChannelFilesCarrier::LogChannelFilesCarrier( const CoreClass* gen, LogChannelFiles *data ):
   CoreCarrier<LogChannelFiles>(gen, data)
{}

LogChannelFilesCarrier::LogChannelFilesCarrier( const LogChannelFilesCarrier& other ):
   CoreCarrier<LogChannelFiles>( other )
   {}

LogChannelFilesCarrier::~LogChannelFilesCarrier()
{}

bool LogChannelFilesCarrier::setProperty( const String &prop, const Item &value )
{
   if( prop == "maxCount" )
      carried()->maxCount( value.forceInteger() );
   else if ( prop == "maxDays" )
      carried()->maxDays( value.forceInteger() );
   else if ( prop == "maxSize" )
      carried()->maxSize( value.forceInteger() );
   else if ( prop == "overwrite" )
      carried()->overwrite( value.isTrue() );
   else if ( prop == "flushAll" )
      carried()->overwrite( value.isTrue() );
   else
      return CoreCarrier<LogChannelFiles>::setProperty( prop, value );

   return true;
}

bool LogChannelFilesCarrier::getProperty( const String &prop, Item &ret ) const
{
   if( prop == "maxCount" )
     ret = (int64) carried()->maxCount();
   else if ( prop == "maxDays" )
     ret = (int64) carried()->maxDays();
   else if ( prop == "maxSize" )
     ret = (int64) carried()->maxSize();
   else if ( prop == "overwrite" )
     ret = carried()->overwrite();
   else if ( prop == "flushAll" )
     ret =  carried()->overwrite();
   else if ( prop == "path" )
     ret = new CoreString(carried()->path());
   else
     return CoreCarrier<LogChannelFiles>::getProperty( prop, ret );

     return true;
}

CoreObject *LogChannelFilesCarrier::clone() const
{
   return new LogChannelFilesCarrier( *this );
}

CoreObject* LogChannelFilesFactory( const CoreClass *cls, void *data, bool )
{
   return new LogChannelFilesCarrier( cls, (LogChannelFiles*) data );
}

}

/* end of logging_mod.cpp */
