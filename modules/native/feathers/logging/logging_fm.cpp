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

#define SRC "modules/native/feathers/logging/logging_ext.cpp"

#include <falcon/fassert.h>
#include <falcon/vmcontext.h>
#include <falcon/string.h>
#include <falcon/function.h>
#include <falcon/item.h>
#include <falcon/itemarray.h>
#include <falcon/itemdict.h>
#include <falcon/stream.h>
#include <falcon/stdhandlers.h>

#include "logging_fm.h"
#include "logging_mod.h"

#include "logarea.h"
#include "logchannel_engine.h"
#include "logchannel_files.h"
#include "logchannel_syslog.h"
#include "logchannel_tw.h"


/*#
   @beginmodule logging
*/

namespace Falcon {
namespace Feathers {

static void internal_log( uint32 level, VMContext* ctx, Function* func, Item* i_message, Item* i_code, LogArea* la = 0 )
{
   if ( i_message == 0 || !i_message->isString()
        || ( i_code != 0 && ! i_code->isOrdinal() ) )
   {
      throw func->paramError( __LINE__, SRC );
   }


   if( la == 0 )
   {
      la = static_cast< LogArea* >( ctx->self().asInst());
   }
   const String& message = *i_message->asString();


   String module, function;
   if( ctx->callDepth() > 1 )
   {
      CallFrame& cf = ctx->callerFrame(1);
      if ( cf.m_function != 0 )
      {
         function = cf.m_function->fullName();
         if( cf.m_function->module() != 0 )
         {
            module = cf.m_function->module()->name();
         }
         else if( cf.m_function->methodOf() != 0 && cf.m_function->methodOf()->module() != 0 )
         {
            module = cf.m_function->methodOf()->module()->name();
         }
      }
   }

   if( i_code != 0 )
   {
      uint32 code = i_code->forceInteger();
      la->log(level, module, function, message, code );

   }
   else {
      la->log( level, module, function, message );
   }

   ctx->returnFrame();
}

// ==============================================
// Class LogArea functions and properties
// ==============================================

namespace CLogArea {
/*#
   @class LogArea
   @brief Collection of log channels.
   @param name The name of the area.
*/

FALCON_DECLARE_FUNCTION( LogArea_init, "name:S" )
FALCON_DEFINE_FUNCTION_P1( LogArea_init )
{
   Item *i_aname = ctx->param(0);

   if ( i_aname == 0 || ! i_aname->isString() )
   {
      throw paramError( __LINE__, SRC );
   }

   LogArea* la = static_cast<LogArea*>(ctx->self().asInst());
   const String& name = *i_aname->asString();
   la->name(name);
   ctx->returnFrame(ctx->self());
}



/*#
   @property minlog LogArea
   @brief Determines what is the minimum log severity active on this area.
   @return A number representing a log severity, or -1

   This property holds the log level accepted by the registered channel
   that is logging the least severe level.

   Notice that severity and numerical values of the logging levels are
   in inverse order. So, the highest severity, which is "fatal", has
   an absolute value of 0, the "error" level has a value of 1 and so on.

   So, to check for the log level you wish to use to be actually streamed
   by some of the registered channel, you have to:

   @code
     if level <= GeneralLog.minlog
        // ok, someone will log my entry
        GeneralLog.log( level, "entry" )
     end
   @endcode

   @see minlog
*/
static void get_minlog( const Class*, const String&, void* instance, Item& value )
{
   LogArea* logArea = static_cast< LogArea* >(instance);
   value.setInteger(logArea->minlog());
}

static void internal_add_remove( Function* func, VMContext* ctx, bool mode )
{
   ModuleLogging* mod = static_cast<ModuleLogging*>(func->methodOf()->module());
   Class* clsChn = mod->classLogChannel();
   Item *i_chn = ctx->param(0);

   if ( i_chn == 0 || ! i_chn->isInstanceOf( clsChn ) )
   {
      throw func->paramError(__LINE__, SRC);
   }

   LogArea* la = static_cast<LogArea*>(ctx->self().asInst());
   Class* cls = 0;
   void* data = 0;
   i_chn->asClassInst(cls,data);
   LogChannel* chn = static_cast<LogChannel*>( cls->getParentData(clsChn, data) );

   if( mode )
   {
      la->addChannel(chn);
   }
   else {
      la->removeChannel(chn);
   }
   ctx->returnFrame();
}

/*#
   @method add LogArea
   @brief Adds a channel to this log area.
   @param channel The channel to be added.
*/
FALCON_DECLARE_FUNCTION( add, "channel:LogChannel" )
FALCON_DEFINE_FUNCTION_P1( add )
{
   internal_add_remove(this,ctx, true);
}

/*#
   @method remove LogArea
   @brief Removes a channel from this log area.
   @param channel The channel to be removed.
*/

FALCON_DECLARE_FUNCTION( remove, "channel:LogChannel" )
FALCON_DEFINE_FUNCTION_P1( remove )
{
   internal_add_remove(this, ctx, false);
}


/*#
   @method log LogArea
   @brief Sends a log entry to all the registered channels.
   @param level The level of the log entry.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   The @b level parameter can be an integer number, or one of the following
   conventional constants (exposed in the module) representing levels:

   - @b LVC: critical; the application met a total failure condition and
              is going to halt.
   - @b LVE: error; the application met an error condition, possibly dangerous
              enough to cause future termination or malfunction, but not
              dangerous enough to justify immediate exit.
   - @b LVW: warning; the application met an unusual condition that that should
              be noted and known by other applications, developers or users
              checking for possible problems.
   - @b LVI: information; the application wants to indicate that a normal or
              expected event actually happened.
   - @b LVD: detail; more detailed information that can be considered superfluous in normal situations.
   - @b LVD0: A message useful to track debugging and development information.
   - @b LVD1: lower debug; debug used for very low level, and specialized debugging.
   - @b LVD2: still lower debug.
*/
FALCON_DECLARE_FUNCTION( log, "level:N,message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( log )
{
   Item *i_level = ctx->param(0);
   Item *i_message = ctx->param(1);
   Item *i_code = ctx->param(2);

   if ( i_level == 0 || ! i_level->isOrdinal() )
   {
      throw paramError( __LINE__, SRC );
   }

   uint32 level = i_level->forceInteger();
   internal_log( level, ctx, this, i_message, i_code );
}

/*#
   @method logc LogArea
   @brief Sends a critical log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logc, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logc )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_C, ctx, this, i_message, i_code );
}

/*#
   @method loge LogArea
   @brief Sends a error level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( loge, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( loge )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_E, ctx, this, i_message, i_code );
}

/*#
   @method logw LogArea
   @brief Sends a warn level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logw, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logw )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_W, ctx, this, i_message, i_code );
}

/*#
   @method logi LogArea
   @brief Sends an information level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logi, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logi )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_I, ctx, this, i_message, i_code );
}

/*#
   @method logd LogArea
   @brief Sends a detail information level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logd, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logd )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_D, ctx, this, i_message, i_code );
}

/*#
   @method logd0 LogArea
   @brief Sends a debug information level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logd0, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logd0 )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_D0, ctx, this, i_message, i_code );
}

/*#
   @method logd1 LogArea
   @brief Sends a detailed debug information level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logd1, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logd1 )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_D1, ctx, this, i_message, i_code );
}

/*#
   @method logd2 LogArea
   @brief Sends a very detailed debug information level log entry to all the registered channels.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   @see LogArea.log
*/
FALCON_DECLARE_FUNCTION( logd2, "message:S,code:[N]" )
FALCON_DEFINE_FUNCTION_P1( logd2 )
{
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);

   internal_log( LOGLEVEL_D2, ctx, this, i_message, i_code );
}
}

// ==============================================
// Class LogArea definition
// ==============================================

ClassLogArea::ClassLogArea():
   Class("LogArea")
{
   init();
}

ClassLogArea::ClassLogArea( const String& name ):
   Class(name)
{
   // no need to add properties for sublcasses
}

void ClassLogArea::init()
{
   setConstuctor( new CLogArea::Function_LogArea_init );

   addProperty( "minlog", &CLogArea::get_minlog );

   addMethod( new CLogArea::Function_add );
   addMethod( new CLogArea::Function_remove );
   addMethod( new CLogArea::Function_log );

   addMethod( new CLogArea::Function_logc );
   addMethod( new CLogArea::Function_loge );
   addMethod( new CLogArea::Function_logw );
   addMethod( new CLogArea::Function_logi );
   addMethod( new CLogArea::Function_logd );
   addMethod( new CLogArea::Function_logd0 );
   addMethod( new CLogArea::Function_logd1 );
   addMethod( new CLogArea::Function_logd2 );
}

ClassLogArea::~ClassLogArea()
{}

void ClassLogArea::dispose( void* instance ) const
{
   LogArea* la = static_cast<LogArea*>(instance);
   la->decref();
}

void* ClassLogArea::clone( void* instance ) const
{
   LogArea* la = static_cast<LogArea*>(instance);
   la->incref();
   return la;
}

void* ClassLogArea::createInstance() const
{
   LogArea* la = new LogArea("unnamed");
   return la;
}


void ClassLogArea::gcMarkInstance( void* instance, uint32 mark ) const
{
   LogArea* la = static_cast<LogArea*>(instance);
   la->gcMark( mark );
}

bool ClassLogArea::gcCheckInstance( void* instance, uint32 mark ) const
{
   LogArea* la = static_cast<LogArea*>(instance);
   return la->currentMark() == mark;
}


// ==============================================
// Singleton GenericLog
// ==============================================


ClassGeneralLogAreaObj::ClassGeneralLogAreaObj( Class* parent ):
    ClassLogArea("%GeneralLog")
{
   setParent(parent);
}

ClassGeneralLogAreaObj::~ClassGeneralLogAreaObj()
{
}


void* ClassGeneralLogAreaObj::createInstance() const
{
   // this is abstract.
   return FALCON_CLASS_CREATE_AT_INIT;
};


bool ClassGeneralLogAreaObj::op_init( VMContext* ctx, void*, int32 ) const
{
   ModuleLogging* mod = static_cast<ModuleLogging*>( module() );
   LogArea* genlog = mod->genericArea();
   genlog->incref();
   ctx->topData() = FALCON_GC_STORE( this, genlog );
   return false;
}


// ==============================================
// Class LogChannel methods and properties
// ==============================================


namespace {


/*#
   @class LogChannel
   @brief Abstract class receiving log requests from log areas.

   This class cannot directly instantiated. Calling it directly
   will generate a code error.
*/


/*#
   @property level LogChannel
   @brief Logging level used on this channel

*/
static void get_level( const Class*, const String&, void* instance, Item& value )
{
   LogChannel* chn = static_cast< LogChannel* >(instance);
   value.setInteger(chn->level());
}

static void set_level( const Class*, const String&, void* instance, const Item& value )
{
   LogChannel* chn = static_cast< LogChannel* >(instance);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }
   chn->level( value.forceInteger());
}


/*#
   @property format LogChannel
   @brief Gets or set the log message formatting setting.

   The message @b format is a template string filled with informations from
   the logging system. Some logging subsystems (as the MS-Windows Event Logger,
   or as the POSIX SYSLOG system) fill autonomously informations on behalf of
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

static void get_format( const Class*, const String&, void* instance, Item& value )
{
   LogChannel* chn = static_cast< LogChannel* >(instance);
   String* string = new String;
   chn->getFormat(*string );
   value = FALCON_GC_HANDLE( string );
}


static void set_format( const Class*, const String&, void* instance, const Item& value )
{
   LogChannel* chn = static_cast< LogChannel* >(instance);
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("S") );
   }
   const String& format = *value.asString();
   chn->setFormat( format );
}
} /* end of namespace CLogChannel */


ClassLogChannel::ClassLogChannel():
         Class("LogChannel")
{
   init();
}

ClassLogChannel::ClassLogChannel( const String& name ):
   Class( name )
{
   // no need to add properties for subclasses
}

ClassLogChannel::~ClassLogChannel()
{
}

void ClassLogChannel::init()
{
   addProperty("format", &get_format, &set_format );
   addProperty("level", &get_level, &set_level );
}


void ClassLogChannel::dispose( void* instance ) const
{
   LogChannel* chn = static_cast<LogChannel*>(instance);
   chn->decref();
}

void* ClassLogChannel::clone( void* ) const
{
   // log channels are uncloneable.
   return 0;
}

void* ClassLogChannel::createInstance() const
{
   // class is pure virtual
   return 0;
}

 void ClassLogChannel::gcMarkInstance( void* instance, uint32 mark ) const
 {
    LogChannel* chn = static_cast<LogChannel*>(instance);
    chn->gcMark( mark );
 }

 bool ClassLogChannel::gcCheckInstance( void* instance, uint32 mark ) const
 {
    LogChannel* chn = static_cast<LogChannel*>(instance);
    return chn->currentMark() >= mark;
 }


// ==============================================
// Class LogChannelStream methods and properties
// ==============================================

namespace CLogChannelStream {
/*#
   @class LogChannelStream
   @brief Logs on an open stream.
   @param stream the text writer where to send the log messages.
   @param initial level the log level.
   @optparam format a format for the log.
   @optparam encoding a text encoding for the log.

   If given, the @b format parameter is used to configure how each log entry
   will look once rendered on the final stream. See @a LogChannel.format for
   a detailed description.

   If not specified, the output format of the text sent to the log will be
   "C" (untranslated).

   @see LogChannel.format
*/
FALCON_DECLARE_FUNCTION( init, "stream:Stream,level:N,format:[S],encoding:[S]" )
FALCON_DEFINE_FUNCTION_P1( init )
{
   Item *i_stream = ctx->param(0);
   Item *i_level = ctx->param(1);
   Item *i_format = ctx->param(2);
   Item *i_encoding = ctx->param(3);

   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   if( i_stream == 0 || ! i_stream->isInstanceOf(clsStream)
       || i_level == 0 || ! i_level->isOrdinal()
       || ( i_format != 0 && ! (i_format->isString()|| i_format->isNil()) )
       || ( i_encoding != 0 && ! i_encoding->isString() )
       )
   {
      throw paramError( __LINE__, SRC );
   }

   Stream* s = static_cast<Stream*>(i_stream->asParentInst( clsStream ) );
   uint32 l = (uint32) i_level->forceInteger();
   String encoding = i_encoding == 0 ? String("C") : *i_encoding->asString();

   Transcoder* tc = Engine::instance()->getTranscoder( encoding );
   if( tc == 0 )
   {
      throw paramError( "Unknown encoding \"" + encoding + "\"", __LINE__, SRC );
   }

   TextWriter* tw = new TextWriter( s, tc );
   LogChannelTW* lcs;
   if( i_format == 0 || i_format->isNil() )
   {
      lcs = new LogChannelTW(tw, l);
   }
   else
   {
      const String& format = *i_format->asString();
      lcs = new LogChannelTW(tw, format, l);
   }

   ctx->self() = FALCON_GC_STORE( methodOf(), lcs );
   ctx->returnFrame(ctx->self());
}

/*#
   @property flushAll LogChannelStream
   @brief Reads or set the flush all mode.

   Stream based channels are usually writing data on buffered streams.
   The default behavior is that of flushing the buffer as soon as a log line is
   written. For some tasks where a large amount of log is written, this may
   be an overkill.
*/

static void get_flushall( const Class*, const String&, void* instance, Item& value )
{
   LogChannelTW* chn = static_cast< LogChannelTW* >(instance);
   value.setBoolean( chn->flushAll() );
}


static void set_flushall( const Class*, const String&, void* instance, const Item& value )
{
   LogChannelTW* chn = static_cast< LogChannelTW* >(instance);
   chn->flushAll( value.isTrue() );
}
} /* end of namespace CLogChannelStream */

ClassLogChannelStream::ClassLogChannelStream( Class* parent ):
         ClassLogChannel( "LogChannelStream")
{
   setParent(parent);

   setConstuctor( new CLogChannelStream::Function_init );
   addProperty("flushall", &CLogChannelStream::get_flushall, &CLogChannelStream::set_flushall );
}

ClassLogChannelStream::~ClassLogChannelStream()
{
}

void* ClassLogChannelStream::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


// ==============================================
// Class LogChannelSyslog methods and properties
// ==============================================

namespace CLogChannelSyslog {

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
   will look once rendered on the final stream. See @a LogChannel.format for
   a detailed description.

   @see LogChannel.format
*/
FALCON_DECLARE_FUNCTION( init, "identity:S, facility:S, level:N,format:[S]" )
FALCON_DEFINE_FUNCTION_P1( init )
{
   Item *i_identity = ctx->param(0);
   Item *i_facility = ctx->param(1);
   Item *i_level = ctx->param(2);
   Item *i_format = ctx->param(3);

   if( i_identity == 0 || ! i_identity->isString()
	   || i_facility == 0 || ! i_facility->isOrdinal()
       || i_level == 0 || ! i_level->isOrdinal()
       || ( i_format != 0 && ! i_format->isString() )
       )
   {
      throw paramError( __LINE__, SRC );
   }

   uint32 f = (uint32) i_facility->forceInteger();
   uint32 l = (uint32) i_level->forceInteger();

   try
   {
      LogChannelSyslog* lc = new LogChannelSyslog(*i_identity->asString(), f, l);

      if( i_format != 0 )
	      lc->setFormat( *i_format->asString() );
   }
   catch( Error* err )
   {
	   err->errorDescription( "Error opening system log" );
	   throw;
   }
}

} /* end of namespace CLogChannelSyslog */

ClassLogChannelSyslog::ClassLogChannelSyslog( Class* parent ):
         ClassLogChannel( "LogChannelSyslog")
{
   setParent(parent);

   setConstuctor( new CLogChannelSyslog::Function_init );
}

ClassLogChannelSyslog::~ClassLogChannelSyslog()
{
}

void* ClassLogChannelSyslog::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


//===============================================
// LogChannelFiles
//===============================================

namespace CLogChannelFiles {
/*#
   @class LogChannelFiles
   @brief Log channel sending logs to a set of (possibly) rotating local files.
   @param path The complete filename were the logs are to be sent.
   @param level Minimum severity level logged by this channel.
   @optparam format Message formatting used by this channel.
   @optparam maxCount Number of maximum log files generated by this channel.
   @optparam maxSize Maximum size of each file.
   @optparam maxDays Maximum days of lifetime for each file.
   @optparam overwrite If true, overwrite the base log file on open, if found.
   @optparam flushAll If false, do NOT flush at every log entry.
   @optparam encoding Text encoding used for output (defaults to "C" = no encoding).

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

   For example, files generated through the path "logs/my_app.%.log" with a
   **maxCount** of 10 will be numbered as (olde()st to newest):
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

FALCON_DECLARE_FUNCTION( init,
         "path:S,level:N,format:[S],maxCount:[N],maxSize:[N],maxDays:[N],overwrite:[B],flushAll:[B],encoding:[S]" )
FALCON_DEFINE_FUNCTION_P1( init )
{
   Item* i_path = ctx->param(0);
   Item* i_level = ctx->param(1);
   Item* i_format = ctx->param(2);
   Item* i_maxCount = ctx->param(3);
   Item* i_maxSize = ctx->param(4);
   Item* i_maxDays = ctx->param(5);
   Item* i_overwrite = ctx->param(6);
   Item* i_flushAll = ctx->param(7);
   Item* i_encodingAll = ctx->param(8);

   if( i_path == 0 || ! i_path->isString()
      || ( i_level != 0 && ! ( i_level->isOrdinal() || i_level->isNil() ))
      || ( i_format != 0 && ! ( i_format->isString() || i_format->isNil() ))
      || ( i_maxCount != 0 && ! ( i_maxCount->isOrdinal() || i_maxCount->isNil() ))
      || ( i_maxSize != 0 && ! ( i_maxSize->isOrdinal() || i_maxSize->isNil() ))
      || ( i_maxDays != 0 && ! ( i_maxDays->isOrdinal() || i_maxDays->isNil() ))
      || ( i_encodingAll != 0 && ! ( i_maxDays->isString() || i_encodingAll->isNil() ))
      )
   {
      throw paramError( __LINE__, SRC );
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

   ctx->self() = FALCON_GC_STORE( methodOf(), lcf );
   ctx->returnFrame( ctx->self() );
}

/*#
   @method open LogChannelFiles
   @brief Opens the stream to the given main log file.
   @raise IoError if the file can't be opened.
*/
FALCON_DECLARE_FUNCTION( open, "" )
FALCON_DEFINE_FUNCTION_P1( open )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( ctx->self().asInst() );
   cc->open();
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
FALCON_DECLARE_FUNCTION( reset, "" )
FALCON_DEFINE_FUNCTION_P1( reset )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( ctx->self().asInst() );
   cc->reset();
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
FALCON_DECLARE_FUNCTION( rotate, "" )
FALCON_DEFINE_FUNCTION_P1( rotate )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( ctx->self().asInst() );
   cc->rotate();
}

/*#
   @property path LogChannelFiles
   @brief Contains the path to the master log file.

   This property is read-only.
*/

static void get_path( const Class*, const String&, void* instance, Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   value = FALCON_GC_HANDLE( new String(cc->path()) );
}


/*#
   @property maxSize LogChannelFiles
   @brief Maximum size of the main log file before being automatically rolled.

   Zero means disabled; that is, the size of the log file is unlimited.
*/
static void get_maxSize( const Class*, const String&, void* instance, Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   value.setInteger( cc->maxSize() );
}

static void set_maxSize( const Class*, const String&, void* instance, const Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }
   cc->maxSize(value.forceInteger());
}

/*#
   @property maxCount LogChannelFiles
   @brief Maximum number of rolled back log files before starting deleting them.

   Zero means that the file is never rolled back; eventually, when hitting a limit,
   it is truncated from the beginning and the log starts all over.
*/

static void get_maxCount( const Class*, const String&, void* instance, Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   value.setInteger( cc->maxCount() );
}

static void set_maxCount( const Class*, const String&, void* instance, const Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }
   cc->maxCount(value.forceInteger());
}

/*#
   @property overwrite LogChannelFiles
   @brief If true, opening a non-empty log will delete it.

   The default for this property is false. This means that the default
   behavior is that to append log entries to the previously existing ones.
*/

static void get_overwrite( const Class*, const String&, void* instance, Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   value.setBoolean( cc->overwrite() );
}

static void set_overwrite( const Class*, const String&, void* instance, const Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   cc->overwrite(value.isTrue());
}


/*#
   @property maxDays LogChannelFiles
   @brief Number of days to keep logging on the master file before rolling it.

   Zero means disabled.
*/


static void get_maxDays( const Class*, const String&, void* instance, Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   value.setInteger( cc->maxDays() );
}

static void set_maxDays( const Class*, const String&, void* instance, const Item& value )
{
   LogChannelFiles* cc = static_cast< LogChannelFiles* >( instance );
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }
   cc->maxDays(value.forceInteger());
}

} /* end of namespace CLogChannelFiles */

ClassLogChannelFiles::ClassLogChannelFiles( Class* parent ):
    ClassLogChannel( "LogChannelFiles")
{
   setParent(parent);

   setConstuctor( new CLogChannelFiles::Function_init );

   addMethod( new CLogChannelFiles::Function_open );
   addMethod( new CLogChannelFiles::Function_reset );
   addMethod( new CLogChannelFiles::Function_rotate );

   addProperty("path", &CLogChannelFiles::get_path );
   addProperty("maxCount", &CLogChannelFiles::get_maxCount, &CLogChannelFiles::set_maxCount );
   addProperty("maxDays", &CLogChannelFiles::get_maxDays, &CLogChannelFiles::set_maxDays );
   addProperty("maxSize", &CLogChannelFiles::get_maxSize, &CLogChannelFiles::set_maxSize );
   addProperty("overwrite", &CLogChannelFiles::get_overwrite, &CLogChannelFiles::set_overwrite );
}

ClassLogChannelFiles::~ClassLogChannelFiles()
{
}

void* ClassLogChannelFiles::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


ClassLogChannelEngine::ClassLogChannelEngine( Class* parent ):
    ClassLogChannel( "LogChannelEngine")
{
   setParent(parent);
}

//=================================================================
// Class log channel engine
//=================================================================

ClassLogChannelEngine::~ClassLogChannelEngine()
{
}

void* ClassLogChannelEngine::createInstance() const
{
   return new LogChannelEngine;
}


bool ClassLogChannelEngine::op_init( VMContext* , void* , int32 ) const
{
   // accept the created instance as-is
   return false;
}

//=================================================================
// Proxy log functions
//=================================================================

namespace {
FALCON_DECLARE_FUNCTION( LOG, "level:N,message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( minlog, "" )
FALCON_DECLARE_FUNCTION( LOGC, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGE, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGW, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGI, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGD, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGD0, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGD1, "message:S,code:[N]" )
FALCON_DECLARE_FUNCTION( LOGD2, "message:S,code:[N]" )

/*#
   @function minlog
   @brief Determines what is the minimum log severity active on the GeneircLog area.
   @return A number representing a log severity, or -1

   This function is actually a shortcut to @a LogArea.minlog applied on @a GeneralLog.
*/
FALCON_DEFINE_FUNCTION_P1(minlog)
{
   ModuleLogging* mod = static_cast<ModuleLogging*>( module() );
   LogArea* genlog = static_cast< LogArea* >( mod->genericArea() );
   ctx->returnFrame( (int64) genlog->minlog() );
}


/*#
   @function log
   @brief Shortcut to log on the generic area.
   @param level The level of the log entry.
   @param message The message to be logged.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area.

   It is provided in this module for improved performance.

   The @b level parameter can be an integer number, or one of the following
   conventional constants representing levels:

   - @b LVC: failure; the application met a total failure condition and
              is going to halt.
   - @b LVE: error; the application met an error condition, possibly dangerous
              enough to cause future termination or malfunction, but not
              dangerous enough to justify immediate exit.
   - @b LVW: warning; the application met an unusual condition that that should
              be noted and known by other applications, developers or users
              checking for possible problems.
   - @b LVI: information; the application wants to indicate that a normal or
              expected event actually happened.
   - @b LVD: detail; More detailed information about a normal event.
   - @b LVD0: debug; A message useful to track debugging and development information.
   - @b LVD1: lower debug; debug used for very low level, and specialized debugging.
   - @b LVD2: still lower debug.
*/


FALCON_DEFINE_FUNCTION_P1(LOG)
{
   ModuleLogging* mod = static_cast<ModuleLogging*>( module() );
   LogArea* genlog = mod->genericArea();

   Item *i_level = ctx->param(0);
   Item *i_message = ctx->param(1);
   Item *i_code = ctx->param(2);

   if ( i_level == 0 || ! i_level->isOrdinal() )
   {
      throw paramError(__LINE__, SRC);
   }

   internal_log( (uint32) i_level->forceInteger(), ctx, this, i_message, i_code, genlog );
}


static void s_genlog( uint32 lvl, VMContext* ctx, Function* func )
{
   ModuleLogging* mod = static_cast<ModuleLogging*>( func->module() );
   Item *i_message = ctx->param(0);
   Item *i_code = ctx->param(1);
   internal_log( lvl, ctx, func, i_message, i_code, mod->genericArea() );
}

/*#
   @function LOGC
   @brief Shortcut to log a critical error on the generic area.
   @param message The message to be logged at fatal level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the fatal error level.

   It is provided in this module for improved performance.
*/

FALCON_DEFINE_FUNCTION_P1(LOGC)
{
   s_genlog( LOGLEVEL_C, ctx, this );
}

/*#
   @function LOGE
   @optparam code A numeric code representing an application specific message ID.
   @brief Shortcut to log an error on the generic area.
   @param message The message to be logged at error level.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the error level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGE)
{
   s_genlog( LOGLEVEL_E, ctx, this );
}


/*#
   @function LOGW
   @brief Shortcut to log a warning on the generic area.
   @param message The message to be logged at warn level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the warn level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGW)
{
   s_genlog( LOGLEVEL_W, ctx, this );
}


/*#
   @function LOGI
   @brief Shortcut to log an information message on the generic area.
   @param message The message to be logged at info level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the info level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGI)
{
   s_genlog( LOGLEVEL_I, ctx, this );
}

/*#
   @function LOGD
   @brief Shortcut to log a detail message on the generic area.
   @param message The message to be logged at debug level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the detail level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGD)
{
   s_genlog( LOGLEVEL_D, ctx, this );
}

/*#
   @function LOGD0
   @brief Shortcut to log a debug message on the generic area.
   @param message The message to be logged at debug level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the debug level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGD0)
{
   s_genlog( LOGLEVEL_D0, ctx, this );
}

/*#
   @function LOGD1
   @brief Shortcut to log a low level debug message on the generic area.
   @param message The message to be logged at debug level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the debug-1 level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGD1)
{
   s_genlog( LOGLEVEL_D1, ctx, this );
}


/*#
   @function LOGD2
   @brief Shortcut to log a low level debug message on the generic area.
   @param message The message to be logged at debug level.
   @optparam code A numeric code representing an application specific message ID.

   This method is equivalent to call @b log on the @a GeneralLog object,
   that is, on the default log area, indicating the debug-2 level.

   It is provided in this module for improved performance.
*/
FALCON_DEFINE_FUNCTION_P1(LOGD2)
{
   s_genlog( LOGLEVEL_D2, ctx, this );
}
}

ModuleLogging::ModuleLogging():
         Module("LoggingModule")
{
   this->addConstant( "LOGFMT_TRACE", "[%s %M.%f]\t%m");
   this->addConstant( "LOGFMT_ERROR", "%T\t%L%C\t[%a]\t%m");
   this->addConstant( "LOGFMT_ERRORP", "%T\t%L%C\t[%a:%M.%f]\t%m");
   this->addConstant( "LOGFMT_ERRORT", "%T\t%L%C\t[%M.%f]\t%m");
   this->addConstant( "LOGFMT_ENTRY", "%T\t(%L) %m");
   this->addConstant( "LOGFMT_ENTRYP", "%T\t(%L) [%a]\t%m");

   this->addConstant( "LVC", LOGLEVEL_C );
   this->addConstant( "LVE", LOGLEVEL_E );
   this->addConstant( "LVW", LOGLEVEL_W );
   this->addConstant( "LVI", LOGLEVEL_I );
   this->addConstant( "LVD", LOGLEVEL_D );
   this->addConstant( "LVD0", LOGLEVEL_D0 );
   this->addConstant( "LVD1", LOGLEVEL_D1 );
   this->addConstant( "LVD2", LOGLEVEL_D2 );
   this->addConstant( "LVALL", LOGLEVEL_ALL );

   m_logArea = new ClassLogArea;
   m_logChannel = new ClassLogChannel;
   m_generalArea = new LogArea("General");

   *this
       << new Function_LOG
       << new Function_LOGC
       << new Function_LOGE
       << new Function_LOGW
       << new Function_LOGI
       << new Function_LOGD
       << new Function_LOGD0
       << new Function_LOGD1
       << new Function_LOGD2

       << m_logArea
       << m_logChannel

       << new ClassLogChannelFiles( m_logChannel )
       << new ClassLogChannelStream( m_logChannel )
       << new ClassLogChannelStream( m_logChannel )
       << new ClassLogChannelEngine( m_logChannel )
   ;

   ClassGeneralLogAreaObj* glog = new ClassGeneralLogAreaObj( m_logArea );
   addObject( glog, false );
}

ModuleLogging::~ModuleLogging()
{
}

}
}

/* end of logging_ext.cpp */
