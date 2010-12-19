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
#include <falcon/corecarrier.h>
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
   @beginmodule feathers.logging
*/

namespace Falcon {
namespace Ext {

static void s_log( LogArea* a, uint32 lev, VMachine* vm, const String& msg, uint32 code )
{
   StackFrame* sf = vm->currentFrame();

   a->log( lev,
         sf->m_module->module()->name(),
         sf->m_symbol->name(),
         msg,
		 code );

}

FALCON_FUNC  GeneralLog_init( ::Falcon::VMachine *vm )
{
   CoreCarrier<LogArea>* cc = static_cast< CoreCarrier<LogArea>* >(vm->self().asObject());
   cc->carried( new LogArea( "general" ) );
}

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

   CoreCarrier<LogArea>* cc = static_cast< CoreCarrier<LogArea>* >(self);
   cc->carried( new LogArea( *i_aname->asString() ) );
}

/*#
   @method minlog LogArea
   @brief Determines what is the minimum log severity active on this area.
   @return A number representing a log severity, or -1

   This function returns the log level accepted by the registered channel
   that is logging the least severe level.

   Notice that severity and numerical values of the logging levels are
   in inverse order. So, the highest severity, which is "fatal", has
   an absolute value of 0, the "error" level has a value of 1 and so on.

   So, to check for the log level you wish to use to be actually streamed
   by some of the registered channel, you have to:

   @code
     if level <= GeneralLog.minlog()
        // ok, someone will log my entry
        GeneralLog.log( level, "entry" )
     end
   @endcode

   @see gminlog
*/
FALCON_FUNC  LogArea_minlog( ::Falcon::VMachine *vm )
{
   CoreCarrier<LogArea>* cc = static_cast< CoreCarrier<LogArea>* >(vm->self().asObject());
   vm->retval( (int64) cc->carried()->minlog() );
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

   CoreCarrier<LogArea>* cc = static_cast< CoreCarrier<LogArea>* >(vm->self().asObject());
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

   CoreCarrier<LogArea>* cc = static_cast< CoreCarrier<LogArea>* >(vm->self().asObject());
   CoreCarrier<LogChannel>* chn = static_cast< CoreCarrier<LogChannel>* >( i_chn->asObjectSafe() );
   cc->carried()->removeChannel( chn->carried() );
}

/*#
   @method log LogArea
   @brief Sends a log entry to all the registred channels.
   @param level The level of the log entry.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   The @b level parameter can be an integer number, or one of the following
   conventional constants representing levels:

   - @b LOGF: failure; the application met a total failure condition and
              is going to halt.
   - @b LOGE: error; the application met an error condition, possibly dangerous
              enough to cause future termination or malfunction, but not
              dangerous enough to justify immediate exit.
   - @b LOGW: warning; the application met an unusual condition that that should
              be noted and known by other applications, developers or users
              checking for possible problems.
   - @b LOGI: infromation; the application wants to indicate that a normal or
              expected event actually happened.
   - @b LOGD: debug; A message useful to track debugging and development information.
   - @b LOGD1: lower debug; debug used for very low level, and specialized debugging.
   - @b LOGD2: still lower debug.
*/
FALCON_FUNC  LogArea_log( ::Falcon::VMachine *vm )
{
   Item *i_level = vm->param(0);
   Item *i_message = vm->param(1);
   Item *i_code = vm->param(2);

   if ( i_level == 0 || ! i_level->isOrdinal()
        || i_message == 0 || !i_message->isString()
		|| ( i_code != 0 && ! i_code->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
			.extra( "N,S,[N]" ) );
   }

   CoreCarrier<LogArea>* cc = static_cast< CoreCarrier<LogArea>* >(vm->self().asObject());
   uint32 code = (uint32) (i_code != 0 ? i_code->forceInteger() : 0);
   s_log( cc->carried(), (uint32) i_level->forceInteger(), vm, *i_message->asString(), code );
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

   CoreCarrier<LogChannel>* cc = (CoreCarrier<LogChannel>*)(vm->self().asObject());

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

   The message @b format is a template string filled with informations from
   the logging system. Some loggin subsystems (as the MS-Windows Event Logger,
   or as the Posix SYSLOG system) fill autonomously informations on behalf of
   the application, while others (file and stream based loggers) require a format
   to print meaningful informations as the timestamp.

   The format string can contain the following escape codes:
   - %a: Application area requesting the log (name of the LogArea).
   - %c: Numeric code as passed in the "code" optional parameter.
   - %C: Numeric code as passed in the "code" optional parameter, zero padded to 5 digits.
   - %d: date in YYYY-MM-DD format.
   - %f: Function calling the log request.
   - %l: Log level as a numeric integer.
   - %L: Letter representing the log level (one character).
   - %M: Name of the module requesting the log.
   - %m: Log message.
   - %R: date in RFC2822 format (as "Sun, 06 Sep 2009 18:16:20 +0200")
   - %s: Milliseconds since the start of the program in seconds and fractions.
   - %S: Milliseconds since the start of the program (absolute).
   - %t: timestamp in HH:MM:SS.mmm format.
   - %T: timestamp in YYYY-MM-DD HH:MM:SS.mmm format.

   - %%: The "%" character.


   For example, the pattern "%t\t%L (%M)\t%m" will print something like
   @code
      13:22:18.344   D (testbind)   Debug message from testbind.
   @endcode

   The module also provides some standard log format code that can be useful
   in the most common situations as constants defined at toplevel. They
   are the following:

   - @b LOGFMT_TRACE: "[%s %M.%f]\t%m" -- this is useful for logs meant to trace
     application progress and perform debug sessions. It indicates how many seconds
     and fractions have passed and the function calling sending the log message.

   - @b LOGFMT_ERROR: "%T\t%L%C\t[%a]\t%m" --  this format indicates the complete
     date and time at which an error took place, the error level and code
     (padded to 5 digits), the area that generated this error and the message.

   - @b LOGFMT_ERRORP: "%T\t%L%C\t[%a:%M.%f]\t%m" -- This format is the same as
     LOGFMT_ERROR, but it adds the name of the module and function that generated
     the log to the area description.

   - @b LOGFMT_ERRORT: "%T\t%L%C\t[%M.%f]\t%m" -- This format is the same as
     LOGFMT_ERRORP, but it doesn't report the log area generating the log.
     Useful if you know that you will be using the general area only.

   - @b LOGFMT_ENTRY: "%T\t(%L) %m" -- Simpler format, reporting the day and time in
     which a log entry is generated, the log level and the message.

   - @b LOGFMT_ENTRYP: "%T\t(%L)  [%a]%m" -- Like LOGFMT_ENTRY, but reporting
     also the log area.
*/
FALCON_FUNC  LogChannel_format( ::Falcon::VMachine *vm )
{
   Item *i_format = vm->param(0);
   CoreCarrier<LogChannel>* cc = (CoreCarrier<LogChannel>*)(vm->self().asObject());

   // always save the level
   CoreString* ret = new CoreString();
   cc->carried()->getFormat( *ret );
   vm->retval( ret );

   if( i_format != 0 )
   {
      if (! i_format->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin(e_orig_runtime)
               .extra( "S" ) );
      }

      // and eventually change it.
      cc->carried()->setFormat( *i_format->asString() );
   }
}

/*#
   @class LogChannelStream
   @brief Logs on an open stream.
   @param stream the stream where to log.
   @param level the log level.
   @optparam format a format for the log.

   If given, the @b format parameter is used to configure how each log entry
   will look once rendered on the final stream. Seel @a LogChannel.format for
   a detailed description.

   @see LogChannel.format
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

   CoreCarrier<LogChannelStream>* cc = (CoreCarrier<LogChannelStream>*)(vm->self().asObject());
   Stream* s = static_cast<Stream*>(i_stream->asObjectSafe()->getFalconData());
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

/*#
   @method flushAll LogChannelStream
   @brief Reads or set the flush all mode.
   @optparam setting True to have the stream flushed at each write.
   @return The current status setting.

   Stream based channels are usually writing data on buffered streams.
   The default behavior is that of flushing the buffer as soon as a log line is
   written. For some tasks where a large amount of log is written, this may
   be an overkill.
*/
FALCON_FUNC  LogChannelStream_flushAll( ::Falcon::VMachine *vm )
{
   Item *i_setting = vm->param(0);

   CoreCarrier<LogChannelStream>* cc = (CoreCarrier<LogChannelStream>*)(vm->self().asObject());

   // always save the level
   vm->retval( cc->carried()->flushAll() );

   if( i_setting != 0 )
   {
      // and eventually change it.
      cc->carried()->flushAll( i_setting->asBoolean() );
   }
}

/*#
   @class LogChannelSyslog
   @brief Logs on the local system logger facility.
   @param identity Name of the application as known by the logging system.
   @param facility Numeric facility or log type code.
   @param level the log level.
   @optparam format a format for the log.

   This class provides a generic logging interface towards the system logger
   facility. On POSIX systems, this is represented by "syslog", and generically
   handled by the SysLog daemon. On MS-Windows systems, messages logged through
   this channel are sent to the Event Logger.

   Identity is the name under which the logging application presents itself to the
   logging system. This is used by the logging system to sort or specially mark
   the messages coming from this application.

   The facility code is a generic application type, and is used to send more relevant
   logs to more visible files. Under MS-Windows, the facility code is sent untranslated
   in the "message category" field of the event logger, and it's application specific.
   In doubt, it's safe to use 0.

   If given, the @b format parameter is used to configure how each log entry
   will look once rendered on the final stream. Seel @a LogChannel.format for
   a detailed description.

   @see LogChannel.format
*/
FALCON_FUNC  LogChannelSyslog_init( ::Falcon::VMachine *vm )
{
   Item *i_identity = vm->param(0);
   Item *i_facility = vm->param(1);
   Item *i_level = vm->param(2);
   Item *i_format = vm->param(3);

   if( i_identity == 0 || ! i_identity->isString()
	   || i_facility == 0 || ! i_facility->isOrdinal()
       || i_level == 0 || ! i_level->isOrdinal()
       || ( i_format != 0 && ! i_format->isString() )
       )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "S,N,N,[S]" ) );
   }

   CoreCarrier<LogChannelSyslog>* cc = (CoreCarrier<LogChannelSyslog>*)(vm->self().asObject());
   uint32 f = (uint32) i_facility->forceInteger();
   uint32 l = (uint32) i_level->forceInteger();

   try
   {
	   cc->carried( new LogChannelSyslog(*i_identity->asString(), f, l) );

       if( i_format != 0 )
	      cc->carried()->setFormat( *i_format->asString() );
   }
   catch( Error* err )
   {
	   err->errorDescription( FAL_STR( msg_log_openres ) );
	  throw;
   }
}


static CoreObject* s_getGenLog( VMachine* vm )
{
   LiveModule* lmod = vm->currentLiveModule();
   if( lmod->userItems().length() == 0 )
   {
      Item* i_genlog = vm->findWKI( "GeneralLog" );
      fassert( i_genlog != 0 );
      fassert( i_genlog->isOfClass( "%GeneralLog" ) );
      lmod->userItems().append( *i_genlog );
      return i_genlog->asObjectSafe();
   }

   return lmod->userItems()[0].asObjectSafe();
}

//===============================================
// LogChannelFiles
//===============================================

/*#
   @class LogChannelFiles
   @brief Log channel sending logs to a set of (possibly) rotating local files.
   @param path The complete filename were the logs are to be sent.
   @optparam level Minimum severity level logged by this channel.
   @optparam format Message formatting used by this channel.
   @optparam maxCount Number of maximum log files generated by this channel.
   @optparam maxSize Maximum size of each file.
   @optparam maxDays Maximum days of lifetime for each file.
   @optparam overwrite If true, overwrite the base log file on open, if found.
   @optparam flushAll If false, do NOT flush at every log entry.

   This log channel is meant to write to one local file, and to possibly swap it
   into spare files when some criterion are met.

   In its basic default configuration, this stream channel works as a
   LogChannelStream provided with a stream open on a local file via InputOutput
   stream (appending at bottom, or eventually creating the file).

   If the @b maxSize parameter is given, the file is rotated (closed and renamed)
   when it grows beyond the given dimension (in bytes).

   If the @b maxDays parameter is given, the file is rotated after a maximum number
   of days since its creation.

   If both @b maxSize and @b maxDays parameters are given, the file is rotated when
   the first boundary is met.

   The action performed when rotating a file is determined by the @b maxCount
   parameter. If it's zero (the default), the log file is simply truncated to 0,
   that is to say, the writing starts anew. Any other value indicates the maximum
   number of back-rotated files created by this channel. Files are rotated by
   adding a numeric suffix to them; the currently active log file is unnumbered,
   the most recent log file is renumbered as 1, the second last most recent is
   numbered 2 up to the value of @b maxCount parameter. When all the positions
   are occupied and a new rotation takes place, the oldest file is deleted.

   The rotated file names are composed of the @b path parameter and a zero
   padded number wide enough to accommodate all the numbers up to @b maxCount; the
   number is generally appended to the @b path through a "." separator, but it is
   possible to override this by passing a path containing a "%" (percent) character,
   which gets substituted by the padded number. In this case, the main file is
   numbered 0.

   For example, files generated through the path "logs/my_app.%.log" with a @b
   maxCount of 10 will be numbered as (oldest to newest):
   @code
   logs/my_app.10.log
   logs/my_app.09.log
   logs/my_app.08.log
   ...
   logs/my_app.01.log
   logs/my_app.00.log
   @endcode

   while files created through the path "logs/my_app.log" would be numbered:
   @code
   logs/my_app.log.10
   logs/my_app.log.09
   logs/my_app.log.08
   ...
   logs/my_app.log.01
   logs/my_app.log
   @endcode

   If the @b overwrite parameter is true, then the the main file will be overwritten
   in case it exists when opening the log.

   Finally, if the @b flushAll parameter is false, then the stream buffer won't
   be flushed after each log operation.

   This log channel doesn't try to create immediately the main file; this is done
   the first time a log request is processed, or explicitly through the
   @a LogChannelFiles.open method. In the first case, a log request advanced
   by a log area may raise an IoError when

   @note All the parameters can be hot-changed in any moment, except for the
   @b path parameter that is read-only.
*/

FALCON_FUNC  LogChannelFiles_init( ::Falcon::VMachine *vm )
{
   Item* i_path = vm->param(0);
   Item* i_level = vm->param(1);
   Item* i_format = vm->param(2);
   Item* i_maxCount = vm->param(3);
   Item* i_maxSize = vm->param(4);
   Item* i_maxDays = vm->param(5);
   Item* i_overwrite = vm->param(6);
   Item* i_flushAll = vm->param(7);

   if( i_path == 0 || ! i_path->isString()
      || ( i_level != 0 && ! ( i_level->isOrdinal() || i_level->isNil() ))
      || ( i_format != 0 && ! ( i_format->isString() || i_format->isNil() ))
      || ( i_maxCount != 0 && ! ( i_maxCount->isOrdinal() || i_maxCount->isNil() ))
      || ( i_maxSize != 0 && ! ( i_maxSize->isOrdinal() || i_maxSize->isNil() ))
      || ( i_maxDays != 0 && ! ( i_maxDays->isOrdinal() || i_maxDays->isNil() ))
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "S,[N],[S],[N],[N],[N],[B],[B]"));
   }

   uint32 level = i_level == 0 ? LOGLEVEL_ALL : (int32) i_level->forceInteger();
   LogChannelFiles* lcf;
   if( i_format == 0 || i_format->isNil() )
      lcf = new LogChannelFiles( *i_path->asString(), level );
   else
      lcf = new LogChannelFiles(  *i_path->asString(), *i_format->asString(), level );

   if (i_maxCount != 0 && ! i_maxCount->isNil() )
      lcf->maxCount( (int32) i_maxCount->forceInteger() );

   if (i_maxSize != 0 && ! i_maxSize->isNil() )
      lcf->maxSize( i_maxSize->forceInteger() );

   if (i_maxDays != 0 && ! i_maxDays->isNil() )
      lcf->maxDays( (int32) i_maxDays->forceInteger() );

   if (i_overwrite != 0 )
      lcf->overwrite( i_overwrite->isTrue() );

   if (i_flushAll != 0 )
      lcf->flushAll( i_flushAll->isTrue() );

   CoreCarrier<LogChannelFiles>* cc = static_cast< CoreCarrier<LogChannelFiles>* >( vm->self().asObject() );
   cc->carried( lcf );
}

/*#
   @method open LogChannelFiles
   @brief Opens the stream to the given main log file.
   @raise IoError if the file can't be opened.
*/
FALCON_FUNC  LogChannelFiles_open( ::Falcon::VMachine *vm )
{
   CoreCarrier<LogChannelFiles>* cc = static_cast< CoreCarrier<LogChannelFiles>* >( vm->self().asObject() );
   cc->carried()->open();
}

/*#
   @method reset LogChannelFiles
   @brief Clears the current log file.
   @raise IoError if the truncate operation files.

   This operation clears the log file without rotating it.
   Some applications, in certain moments, know that the previous
   log is useless and to be discarded, for example because they
   want to debug the operations from that point on.

   This method also forces do discard all the pending messages
   that was queued for logging but still unwritten.
*/
FALCON_FUNC  LogChannelFiles_reset( ::Falcon::VMachine *vm )
{
   CoreCarrier<LogChannelFiles>* cc = static_cast< CoreCarrier<LogChannelFiles>* >( vm->self().asObject() );
   cc->carried()->reset();
}

/*#
   @method rotate LogChannelFiles
   @brief Rotates the current log file.

   This operation perform a rotation operation, respecting
   the default behavior specified through the @a LogChannelFiles.maxCount
   property, as if one of the limits were hit.

   The operation is actually queued after all pending messages;
   new messages posted after this operation will go to the new
   main file after the rotation, while the pending messages will be
   written to the new log stream.

   @note A failure in the rotate operation will cause the operation
   to silently fail and the log to keep being sent to the current
   log file.

*/
FALCON_FUNC  LogChannelFiles_rotate( ::Falcon::VMachine *vm )
{
   CoreCarrier<LogChannelFiles>* cc = static_cast< CoreCarrier<LogChannelFiles>* >( vm->self().asObject() );
   cc->carried()->rotate();
}

/*#
   @property flushAll LogChannelFiles
   @brief When true, all the log operations cause an immediate flush of the
   underlying stream.
*/


/*#
   @property path LogChannelFiles
   @brief Contains the path to the master log file.

   This property is read-only.
*/

/*#
   @property maxSize LogChannelFiles
   @brief Maximum size of the main log file before being automatically rolled.

   Zero means disabled; that is, the size of the log file is unlimited.
*/

/*#
   @property maxCount LogChannelFiles
   @brief Maximum number of rolled back log files before starting deleting them.

   Zero means that the file is never rolled back; eventually, when hitting a limit,
   it is truncated from the beginning and the log starts all over.
*/

/*#
   @property overwrite LogChannelFiles
   @brief If true, opening a non-empty log will delete it.

   The default for this property is false. This means that the default
   behavior is that to append log entries to the previously existing ones.
*/


/*#
   @property maxDays LogChannelFiles
   @brief Number of days to keep logging on the master file before rolling it.

   Zero means disabled.
*/

//=================================================================
// Proxy log functions
//=================================================================

/*#
   @function gminlog
   @brief Determines what is the minimum log severity active on the GeneircLog area.
   @return A number representing a log severity, or -1

   This function is actually a shortcut to @a LogArea.minlog applied on @a GeneralLog.
*/
FALCON_FUNC  gminlog( ::Falcon::VMachine *vm )
{
   LogArea* genlog = static_cast< CoreCarrier<LogArea>* >( s_getGenLog(vm) )->carried();
   vm->retval( (int64) genlog->minlog() );
}


/*#
   @function glog
   @brief Shortcut to log on the generic area.
   @param level The level of the log entry.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area.

   It is provided in this module for improved performance.

   The @b level parameter can be an integer number, or one of the following
   conventional constants representing levels:

   - @b LOGF: failure; the application met a total failure condition and
              is going to halt.
   - @b LOGE: error; the application met an error condition, possibly dangerous
              enough to cause future termination or malfunction, but not
              dangerous enough to justify immediate exit.
   - @b LOGW: warning; the application met an unusual condition that that should
              be noted and known by other applications, developers or users
              checking for possible problems.
   - @b LOGI: infromation; the application wants to indicate that a normal or
              expected event actually happened.
   - @b LOGD: debug; A message useful to track debugging and development information.
   - @b LOGD1: lower debug; debug used for very low level, and specialized debugging.
   - @b LOGD2: still lower debug.

*/

FALCON_FUNC  glog( ::Falcon::VMachine *vm )
{
   Item *i_level = vm->param(0);
   Item *i_message = vm->param(1);
   Item *i_code = vm->param(2);

   if ( i_level == 0 || ! i_level->isOrdinal()
        || i_message == 0 || !i_message->isString()
		|| (i_code != 0 && ! i_code->isOrdinal() )
		)
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "N,S" ) );
   }

   LogArea* genlog = static_cast< CoreCarrier<LogArea>* >( s_getGenLog(vm) )->carried();
   uint32 code = (uint32)( i_code == 0 ? 0 : i_code->forceInteger() );
   s_log( genlog, (uint32) i_level->forceInteger(), vm, *i_message->asString(), code );

}

void s_genericLog( VMachine* vm, uint32 level )
{
   Item *i_message = vm->param(0);
   Item *i_code = vm->param(1);

   if ( i_message == 0 || ! i_message->isString()
	    || (i_code != 0 && ! i_code->isOrdinal() )
	   )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "S,[N]" ) );
   }

   LogArea* genlog = static_cast< CoreCarrier<LogArea>* >( s_getGenLog(vm) )->carried();
   uint32 code = (uint32)( i_code == 0 ? 0 : i_code->forceInteger() );
   s_log( genlog, level, vm, *i_message->asString(), code );
}

/*#
   @function glogf
   @brief Shortcut to log a fatal error on the generic area.
   @param message The message to be logged at fatal level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the fatal error level.

   It is provided in this module for improved performance.
*/

FALCON_FUNC  glogf( ::Falcon::VMachine *vm )
{
   s_genericLog( vm, LOGLEVEL_FATAL );
}

/*#
   @function gloge
   @optparam code A numeric code representing an application specific message ID.
   @brief Shortcut to log an error on the generic area.
   @param message The message to be logged at error level.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the error level.

   It is provided in this module for improved performance.
*/

FALCON_FUNC  gloge( ::Falcon::VMachine *vm )
{
   s_genericLog( vm, LOGLEVEL_ERROR );
}

/*#
   @function glogw
   @brief Shortcut to log a warning on the generic area.
   @param message The message to be logged at warn level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the warn level.

   It is provided in this module for improved performance.
*/
FALCON_FUNC  glogw( ::Falcon::VMachine *vm )
{
   s_genericLog( vm, LOGLEVEL_WARN );
}

/*#
   @function glogi
   @brief Shortcut to log an information message on the generic area.
   @param message The message to be logged at info level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the info level.

   It is provided in this module for improved performance.
*/
FALCON_FUNC  glogi( ::Falcon::VMachine *vm )
{
   s_genericLog( vm, LOGLEVEL_INFO );
}

/*#
   @function glogd
   @brief Shortcut to log a debug message on the generic area.
   @param message The message to be logged at debug level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the debug level.

   It is provided in this module for improved performance.
*/
FALCON_FUNC  glogd( ::Falcon::VMachine *vm )
{
   s_genericLog( vm, LOGLEVEL_DEBUG );
}


}
}

/* end of logging_ext.cpp */
