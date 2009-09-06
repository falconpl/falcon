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
#include "logging_ext.h"
#include "logging_mod.h"
#include "logging_st.h"

#include "version.h"


namespace Falcon {

template class CoreCarrier<LogArea>;
template class CoreCarrier<LogChannel>;
template class CoreCarrier<LogChannelStream>;

template CoreObject* CoreCarrier_Factory<LogArea>( const CoreClass *cls, void *data, bool );
template CoreObject* CoreCarrier_Factory<LogChannel>( const CoreClass *cls, void *data, bool );
template CoreObject* CoreCarrier_Factory<LogChannelStream>( const CoreClass *cls, void *data, bool );

}

/*#
   @module feather_logging Logging support
   @brief Multithread enabled logging facility.

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

   //====================================
   // General log area

   /*# @object GeneralLog
       @from LogArea
       @brief General logging area.

       This is the default log area used by the @a log function.
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
   // Generic log function
   //
   self->addExtFunc( "glog", &Falcon::Ext::glog )->
      addParam( "level" )->addParam( "message" );
   self->addExtFunc( "glogf", &Falcon::Ext::glogf )->addParam( "message" );
   self->addExtFunc( "gloge", &Falcon::Ext::gloge )->addParam( "message" );
   self->addExtFunc( "glogw", &Falcon::Ext::glogw )->addParam( "message" );
   self->addExtFunc( "glogi", &Falcon::Ext::glogi )->addParam( "message" );
   self->addExtFunc( "glogd", &Falcon::Ext::glogd )->addParam( "message" );

   return self;
}

/* end of logging.cpp */

