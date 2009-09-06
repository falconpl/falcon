/*
   FALCON - The Falcon Programming Language.
   FILE: logging_ext.cpp

   Falcon VM interface to logging module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to configuration parser module.
*/


#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/lineardict.h>
#include <falcon/stream.h>
#include <falcon/memory.h>

#include "logging_ext.h"
#include "logging_mod.h"
#include "logging_st.h"

/*#
   @beginmodule feather_logger
*/

namespace Falcon {
namespace Ext {

// ==============================================
// Class LogArea
// ==============================================

/*#
   @class LogArea
   @brief Collection of log channels.
   @param name The name of the area.
*/

FALCON_FUNC  LogArea_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_aname = vm->param(0);

   if ( i_aname == 0 || ! i_aname->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "S" ) );
   }

   CoreCarrier<LogArea>* cc = dyncast< CoreCarrier<LogArea>* >(self);
   cc->carried( new LogArea( *i_aname->asString() ) );
}

/*#
   @method add LogArea
   @brief Adds a channel to this log area.
   @param channel The channel to be added.
*/
FALCON_FUNC  LogArea_add( ::Falcon::VMachine *vm )
{
   Item *i_chn = vm->param(0);

   if ( i_chn == 0 || ! i_chn->isOfClass( "LogChannel" ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "LogChannel" ) );
   }

   CoreCarrier<LogArea>* cc = dyncast< CoreCarrier<LogArea>* >(vm->self().asObject());
   CoreCarrier<LogChannel>* chn = static_cast< CoreCarrier<LogChannel>* >( i_chn->asObjectSafe() );
   cc->carried()->addChannel( chn->carried() );
}

/*#
   @method remove LogArea
   @brief Removes a channel from this log area.
   @param channel The channel to be removed.
*/

FALCON_FUNC  LogArea_remove( ::Falcon::VMachine *vm )
{
   Item *i_chn = vm->param(0);

   if ( i_chn == 0 || ! i_chn->isOfClass( "LogChannel" ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "LogChannel" ) );
   }

   CoreCarrier<LogArea>* cc = dyncast< CoreCarrier<LogArea>* >(vm->self().asObject());
   CoreCarrier<LogChannel>* chn = dyncast< CoreCarrier<LogChannel>* >( i_chn->asObjectSafe() );
   cc->carried()->removeChannel( chn->carried() );
}

/*#
   @method log LogArea
   @brief Sends a log entry to all the registred channels.
   @param level The level of the log entry.
   @param message The message to be logged.
*/
FALCON_FUNC  LogArea_log( ::Falcon::VMachine *vm )
{
   Item *i_level = vm->param(0);
   Item *i_message = vm->param(1);

   if ( i_level == 0 || ! i_level->isOrdinal()
        || i_message == 0 || !i_message->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "N,S" ) );
   }

   CoreCarrier<LogArea>* cc = dyncast< CoreCarrier<LogArea>* >(vm->self().asObject());

   cc->carried()->log( i_level->forceInteger(), *i_message->asString() );
}

/*#
   @class LogChannel
   @brief Abstract class receiving log requests from log areas.

   This class cannot directly instantiated. Calling it directlty
   will generate a code error.
*/
FALCON_FUNC  LogChannel_init( ::Falcon::VMachine *vm )
{
   CoreObject* self = vm->self().asObject();

   if( self->generator()->symbol()->name() == "LogChannel" )
   {
      throw new CodeError( ErrorParam( e_noninst_cls, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "LogChannel") );
   }
   // otherwise, we have nothing to do.
}

/*#
   @method level LogChannel
   @brief Gets or set the log level.
   @optparam level a new log level to be set.
   @return The current log level.

*/
FALCON_FUNC  LogChannel_level( ::Falcon::VMachine *vm )
{
   Item *i_level = vm->param(0);

   CoreCarrier<LogChannel>* cc = dyncast< CoreCarrier<LogChannel>* >(vm->self().asObject());

   // always save the level
   vm->retval( (int64) cc->carried()->level() );

   if( i_level != 0 )
   {
      if (! i_level->isOrdinal() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin(e_orig_runtime)
               .extra( "N" ) );
      }

      // and eventually change it.
      cc->carried()->level( (uint32) i_level->forceInteger() );
   }
}

/*#
   @method format LogChannel
   @brief Gets or set the log message formatting setting.
   @optparam format the new format to set (a string).
   @return The current log message format (a string).

*/
FALCON_FUNC  LogChannel_format( ::Falcon::VMachine *vm )
{
   Item *i_format = vm->param(0);
   CoreCarrier<LogChannel>* cc = dyncast< CoreCarrier<LogChannel>* >(vm->self().asObject());

   // always save the level
   vm->retval( new CoreString( cc->carried()->format() ) );

   if( i_format != 0 )
   {
      if (! i_format->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin(e_orig_runtime)
               .extra( "S" ) );
      }

      // and eventually change it.
      cc->carried()->format( *i_format->asString() );
   }
}

/*#
   @class LogChannelStream
   @brief Logs on an open stream.
   @param stream the stream where to log.
   @param level the log level.
   @optparam format a format for the log.
*/
FALCON_FUNC  LogChannelStream_init( ::Falcon::VMachine *vm )
{
   Item *i_stream = vm->param(0);
   Item *i_level = vm->param(1);
   Item *i_format = vm->param(2);

   if( i_stream == 0 || ! i_stream->isOfClass( "Stream" )
       || i_level == 0 || ! i_level->isOrdinal()
       || ( i_format != 0 && ! i_format->isString() )
       )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "Stream,N,[S]" ) );
   }

   CoreCarrier<LogChannelStream>* cc = dyncast< CoreCarrier<LogChannelStream>* >(vm->self().asObject());
   Stream* s = dyncast<Stream*>(i_stream->asObjectSafe()->getFalconData());
   uint32 l = (uint32) i_level->forceInteger();

   if( i_format == 0 )
   {
      cc->carried( new LogChannelStream(static_cast<Stream*>(s->clone()), l) );
   }
   else
   {
      cc->carried( new LogChannelStream(static_cast<Stream*>(s->clone()), *i_format->asString(), l) );
   }
}


}
}

/* end of logging_ext.cpp */
