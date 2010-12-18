/*
   FALCON - The Falcon Programming Language.
   FILE: logging.cpp

   The logging module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The confparser module - main file.
*/
#include <falcon/module.h>
#include <falcon/corecarrier.h>
#include "logging_ext.h"
#include "logging_mod.h"
#include "logging_st.h"

#include "version.h"

static Falcon::LogService s_theLogService;

namespace Falcon {

template class CoreCarrier<LogArea>;
template class CoreCarrier<LogChannel>;
template class CoreCarrier<LogChannelStream>;
template class CoreCarrier<LogChannelSyslog>;

template CoreObject* CoreCarrier_Factory<LogArea>( const CoreClass *cls, void *data, bool );
template CoreObject* CoreCarrier_Factory<LogChannel>( const CoreClass *cls, void *data, bool );
template CoreObject* CoreCarrier_Factory<LogChannelStream>( const CoreClass *cls, void *data, bool );
template CoreObject* CoreCarrier_Factory<LogChannelSyslog>( const CoreClass *cls, void *data, bool );

}

/*#
   @module feathers_logging logging
   @brief Multithread enabled logging facility.
   @inmodule feathers

   The @b logging module offers a very advanced facility for logging application
   messages to various device.

   It is based on a Area/Channel architecture.

   The @a LogArea class provides a logical
   subdivision of the semantic meaning of a log entry; sending a log to a certain
   area means that the entry has as certain meaning and function in the application
   architecture. For example, an application can have "security", "trace" and
   "status" areas; security area is meant to log informations and errors concerning
   security policy operations (login, logout, password change and so on). The status
   area may be dedicated to log internal operation status, as, for example failures
   in opening a database, performed queries or amount of records changed. The trace
   area may be specifically dedicated to debug or deep audit.

   Channels are physical or logical device managers, receiving log requests and
   delivering them on final media, rendered as their internal rules dictate.

   The user application posts a log entry to an area, and the area dispatches
   the entry to all the channels that are registered with. Channels can be
   registered with multiple areas, and they can be dynamically added to new
   areas or removed from areas they are currently connected to.

   Each message it's associated with a "level", also called severity, which
   determines how important the log entry is. Channels can be configured to
   perform log operations only to messages having at least a minimum level,
   that is, severe at least as a certain level. Lower level entries are ignored
   by the channel.

   The current logger module provides one default area, called @a GeneralLog, and
   three log channel classes:
   - @a LogChannelStream - writes entries on a Falcon stream separately opened.
   - @a LogChannelSyslog - sends entries to the system log facility. On POSIX
     systems, logs are sent to the Syslog daemon, while on MS-Windows systems
     they are sent to the Event Logger.
   - @a LogChannelFiles - Writes entries to a local file, and swaps or rotates
     the log file on need.

   @section feather_logging_performance Performance considerations

   The logger module is thought to perform the final part of the log operations
   in the most efficient way possible. Each channel performs the final log
   rendering and the dispatching on the underlying media on a separate thread,
   so the operation of generating a log for a certain channel is virtually
   non-blocking and relatively fast.

   Log operations involving fixed parameters are nearly no-ops, as in the following
   example:

   @code
   glog( 2000, "A very low priority level, unlikely to be ever logged" )
   @endcode

   Although there is a cost in calling the @a glog function (logging on the GenericLog
   area), a modern computer can perform such calls in the order of about five millions
   per second.

   However, consider the nature of falcon as a Virtual Machine interpreted scripting
   language. Creating a complete log message may be an heavy operation by itself;
   for example:
   @code

      rendered = ""
      for person in visitors
          rendered += person
          formiddle: rendered += ", "
      end

      GenericLog.log( LOGI, "Today we had " + visitors.len()
          + " visitors, named " + rendered )
   @endcode

   In this case, creating the "rendered" log-friendly representation of the visitors
   is quite slow, and so it's creating the log entry in the log() call.

   If there isn't any channel registered with the GenericLog area, the message will
   be discarded, but the heaviest part of the job has already be done, and in waste.

   In case logs are heavy and frequent, it is useful to wrap log generation of
   the most intensive entries in runtime checks, or better, compile time directives
   that prevent calling logs when the application knows they are actually not wanted
   in the current context.

   The @a LogArea.minlog method returns the minimum log level that is accepted
   by a registered channel (the @a gminlog function operates on the GeneralLog area),
   providing a simple way to prevent logging useless data in runtime:

   @code
      // want debug?
      if LOGD <= gminlog()
         // prepare an heavy log...
         glogd( "Debug heavy log: " + ... )
      end
   @endcode

   @note Log levels appares in inverse order of severity; so LOGF (fatal) is a level
   numerically less than LOGD (debug).

   @section feather_logging_service Service model

   Falcon logging module is exposed to embedding application as a "service".
   This means that embedding applications using the Falcon engine can dynamically
   load the logging module and access the same functionalities available to
   Falcon scripts directly from a C++ SDK interface.

   Actually, LogChannel class hierarchy is directly visible in the scripts; so
   embedding application could create log areas or ever log channels pointing
   back to them, or sharing channels with the scripts to write joint logs (maybe
   from different areas).

   @section feather_logging_extending Extending classes.

   All the classes in this module are highly reflective, and operate on the inner
   C++ classes for performance reason. Because of this, overloading a LogChannel
   class and overloading its @b log method won't cause the LogArea to call back
   the new log method provided by the script; the underlying C++ log method will
   still be called.

   However, it is legal to extend the LogChannel classes; the overloaded log
   method can be used directly by the script even if it will be ignored by
   LogArea instances.
*/

/*#
   @beginmodule feathers_logging
*/
FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // setup DLL engine common data

   Falcon::Module *self = new Falcon::Module();
   self->name( "logging" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "logging_st.h"

   //====================================
   // Some general constants
   //
   self->addConstant( "LOGFMT_TRACE", "[%s %M.%f]\t%m");
   self->addConstant( "LOGFMT_ERROR", "%T\t%L%C\t[%a]\t%m");
   self->addConstant( "LOGFMT_ERRORP", "%T\t%L%C\t[%a:%M.%f]\t%m");
   self->addConstant( "LOGFMT_ERRORT", "%T\t%L%C\t[%M.%f]\t%m");
   self->addConstant( "LOGFMT_ENTRY", "%T\t(%L) %m");
   self->addConstant( "LOGFMT_ENTRYP", "%T\t(%L) [%a]\t%m");

   //====================================
   // Class Log Area
   //

   Falcon::Symbol *c_logarea = self->addClass( "LogArea", &Falcon::Ext::LogArea_init )
         ->addParam("name");
   c_logarea->getClassDef()->factory( &Falcon::CoreCarrier_Factory<Falcon::LogArea> );

   self->addClassMethod( c_logarea, "add", &Falcon::Ext::LogArea_add ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_logarea, "remove", &Falcon::Ext::LogArea_remove ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_logarea, "log", &Falcon::Ext::LogArea_log ).asSymbol()->
      addParam("level")->addParam("message");
   self->addClassMethod( c_logarea, "minlog", &Falcon::Ext::LogArea_minlog );

   //====================================
   // General log area

   /*# @object GeneralLog
       @from LogArea
       @brief General logging area.

       This is the default log area used by the @a glog function.
   */
   Falcon::Symbol *o_genlog = self->addSingleton( "GeneralLog", &Falcon::Ext::GeneralLog_init, true );
   o_genlog->getInstance()->getClassDef()->addInheritance( new Falcon::InheritDef( c_logarea) );
   o_genlog->setWKS( true );

   //====================================
   // Class LogChannel

   // Init prevents direct initialization. -- it's an abstract class.
   Falcon::Symbol *c_logc = self->addClass( "LogChannel", &Falcon::Ext::LogChannel_init );

   self->addClassMethod( c_logc, "level", &Falcon::Ext::LogChannel_level ).asSymbol()->
      addParam("level");
   self->addClassMethod( c_logc, "format", &Falcon::Ext::LogChannel_format ).asSymbol()->
      addParam("format");

   //====================================
   // Class LogChannelStream
   //
   Falcon::Symbol *c_logcs = self->addClass( "LogChannelStream", &Falcon::Ext::LogChannelStream_init )
         ->addParam("level")->addParam("format");
   c_logcs->getClassDef()->factory( &Falcon::CoreCarrier_Factory<Falcon::LogChannelStream> );
   c_logcs->getClassDef()->addInheritance( new Falcon::InheritDef(c_logc) );

   self->addClassMethod( c_logcs, "flushAll", &Falcon::Ext::LogChannelStream_flushAll ).asSymbol()->
      addParam("setting");

   //====================================
   // Class LogChannelSyslog
   //
   Falcon::Symbol *c_logsyslog = self->addClass( "LogChannelSyslog", &Falcon::Ext::LogChannelSyslog_init )
         ->addParam("identity")->addParam("facility")->addParam("level")->addParam("format");
   c_logsyslog->getClassDef()->factory( &Falcon::CoreCarrier_Factory<Falcon::LogChannelSyslog> );
   c_logsyslog->getClassDef()->addInheritance( new Falcon::InheritDef(c_logc) );

   //====================================
   // Class LogChannelFiles
   //
   Falcon::Symbol *c_logfiles = self->addClass( "LogChannelFiles", &Falcon::Ext::LogChannelFiles_init )
         ->addParam("path")->addParam("level")->addParam("format")
         ->addParam("maxCount")->addParam("maxSize")->addParam("maxDays")
         ->addParam("overwrite")->addParam("flushAll");

   c_logfiles->getClassDef()->factory( &Falcon::LogChannelFilesFactory );
   c_logfiles->getClassDef()->addInheritance( new Falcon::InheritDef(c_logc) );

   self->addClassMethod( c_logfiles, "open", &Falcon::Ext::LogChannelFiles_open ).
      setReadOnly(true);
   self->addClassProperty( c_logfiles, "flushAll" );
   self->addClassProperty( c_logfiles, "maxSize" );
  self->addClassProperty( c_logfiles, "maxCount" );
   self->addClassProperty( c_logfiles, "maxDays" );
   self->addClassProperty( c_logfiles, "path" );
   self->addClassProperty( c_logfiles, "overwrite" );

   //====================================
   // Generic log function
   //
   self->addExtFunc( "gminlog", &Falcon::Ext::gminlog );
   self->addExtFunc( "glog", &Falcon::Ext::glog )->
      addParam( "level" )->addParam( "message" );
   self->addExtFunc( "glogf", &Falcon::Ext::glogf )->addParam( "message" );
   self->addExtFunc( "gloge", &Falcon::Ext::gloge )->addParam( "message" );
   self->addExtFunc( "glogw", &Falcon::Ext::glogw )->addParam( "message" );
   self->addExtFunc( "glogi", &Falcon::Ext::glogi )->addParam( "message" );
   self->addExtFunc( "glogd", &Falcon::Ext::glogd )->addParam( "message" );

   //=====================================
   // Constants
   //
   self->addConstant( "LOGF", (Falcon::int64) LOGLEVEL_FATAL );
   self->addConstant( "LOGE", (Falcon::int64) LOGLEVEL_ERROR );
   self->addConstant( "LOGW", (Falcon::int64) LOGLEVEL_WARN );
   self->addConstant( "LOGI", (Falcon::int64) LOGLEVEL_INFO );
   self->addConstant( "LOGD", (Falcon::int64) LOGLEVEL_DEBUG );
   self->addConstant( "LOGD1", (Falcon::int64) LOGLEVEL_D1 );
   self->addConstant( "LOGD2", (Falcon::int64) LOGLEVEL_D2 );

   //======================================
   // Subscribe the service
   //
   self->publishService( &s_theLogService );

   return self;
}

/* end of logging.cpp */

