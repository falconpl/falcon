/*
   The Falcon Programming Language
   FILE: dbus_ext.cpp

   Falcon - DBUS official binding
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 17 Dec 2008 20:12:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: Giancarlo Niccolai

      Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

      http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include "dbus_ext.h"
#include "dbus_st.h"
#include "dbus_mod.h"

#include "version.h"

/*#
   @main The DBus Falcon Module

   DBus is an inter-application remote procedure call protocol that was designed
   specifically to write interoperating applications in desktop environments,
   and across different base technology. It was initially a joint effort of the
   GNome and KDE desktop environments work groups, and it's now available on a wide
   set of systems.

   Although it was born as an alternative windowed application remote control system,
   it is now used on many non-graphical applications as a more standardized and
   general mean to pass data and invoke other application's functionalities.

   The Falcon-DBus module provide Falcon scripts with functions to connect to DBus
   server (that is, applications providing some functions), and to create a DBus
   server able to fulfil requests received from other applications.

   @note This version exposes only client-side functions to access DBus server.
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Mod::DBusModule();
   self->name( "dbus" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "dbus_st.h"

   //============================================================
   // Create the dbus class
   //
   Falcon::Symbol* dbus_cls = self->addClass( "DBus", Falcon::Ext::DBus_init );
   self->addClassMethod( dbus_cls, "signal", Falcon::Ext::DBus_signal ).asSymbol()->
      addParam("path")->addParam("interface")->addParam("name");
   self->addClassMethod( dbus_cls, "invoke", Falcon::Ext::DBus_invoke ).asSymbol()->
      addParam("destination")->addParam("path")->addParam("interface")->addParam("name");
   self->addClassMethod( dbus_cls, "dispatch", Falcon::Ext::DBus_dispatch ).asSymbol()->
      addParam("timeout");
   self->addClassMethod( dbus_cls, "popMessage", Falcon::Ext::DBus_popMessage );
   self->addClassMethod( dbus_cls, "addMatch", Falcon::Ext::DBus_addMatch ).asSymbol()->
      addParam("rule");
   self->addClassMethod( dbus_cls, "removeMatch", Falcon::Ext::DBus_removeMatch ).asSymbol()->
      addParam("rule");
   self->addClassMethod( dbus_cls, "requestName", Falcon::Ext::DBus_requestName ).asSymbol()->
      addParam( "name" )->addParam( "flags" );
   self->addClassMethod( dbus_cls, "addFilter", Falcon::Ext::DBus_addFilter ).asSymbol()->
      addParam( "interface" )->addParam( "name" )->addParam( "handler" )->addParam( "isSignal" );
   self->addClassMethod( dbus_cls, "startDispatch", Falcon::Ext::DBus_startDispatch );
   self->addClassMethod( dbus_cls, "stopDispatch", Falcon::Ext::DBus_stopDispatch );
      
   //============================================================
   // The pending class.
   //
   Falcon::Symbol* dbusp_cls = self->addClass( "%DBusPendingCall"); 
   dbusp_cls->exported( false );
   dbusp_cls->setWKS( true );
   
   self->addClassMethod( dbusp_cls, "wait", Falcon::Ext::DBusPendingCall_wait );
   self->addClassMethod( dbusp_cls, "cancel", Falcon::Ext::DBusPendingCall_cancel );
   self->addClassMethod( dbusp_cls, "completed", Falcon::Ext::DBusPendingCall_completed ).asSymbol()->
      addParam("dispatch");
      
   //============================================================
   // message class
   //
   Falcon::Symbol *dbusmsg_cls = self->addClass( "%DBusMessage" );
   dbusmsg_cls->exported( false );
   dbusmsg_cls->setWKS( true );
   
   self->addClassMethod( dbusmsg_cls, "getDestination", Falcon::Ext::DBusMessage_getDestination );
   self->addClassMethod( dbusmsg_cls, "getSender", Falcon::Ext::DBusMessage_getSender );
   self->addClassMethod( dbusmsg_cls, "getPath", Falcon::Ext::DBusMessage_getPath );
   self->addClassMethod( dbusmsg_cls, "getInterface", Falcon::Ext::DBusMessage_getInterface );
   self->addClassMethod( dbusmsg_cls, "getMember", Falcon::Ext::DBusMessage_getMember );
   self->addClassMethod( dbusmsg_cls, "getArgs", Falcon::Ext::DBusMessage_getArgs );
   
   //============================================================
   // Error for DBUS related ops
   //
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *dbierr_cls = self->addClass( "DBusError", Falcon::Ext::DBusError_init )->
      addParam( "code" )->addParam( "desc" )->addParam( "extra" );
   dbierr_cls->setWKS( true );
   dbierr_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );
   
   return self;
}

/* end of dbus.cpp */
