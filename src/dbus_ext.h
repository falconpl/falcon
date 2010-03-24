/*
   The Falcon Programming Language
   FILE: dbus_ext.cpp

   Falcon - DBUS official binding
   Interface extension functions
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
   Falcon - DBUS official binding
   Interface extension functions - header file
*/

#ifndef dbus_ext_H
#define dbus_ext_H

#include <falcon/module.h>
#include <falcon/error.h>

#ifndef FALCON_ERROR_DBUS_BASE
#define FALCON_ERROR_DBUS_BASE            2300
#endif

namespace Falcon {
namespace Ext {
   FALCON_FUNC  DBus_init( VMachine *vm );
   FALCON_FUNC  DBus_signal( VMachine *vm );
   FALCON_FUNC  DBus_invoke( VMachine *vm );
   FALCON_FUNC  DBus_dispatch( VMachine *vm );
   FALCON_FUNC  DBus_popMessage( VMachine *vm );
   FALCON_FUNC  DBus_addMatch( VMachine *vm );
   FALCON_FUNC  DBus_removeMatch( VMachine *vm );
   FALCON_FUNC  DBus_requestName( VMachine *vm );
   FALCON_FUNC  DBus_addFilter( VMachine *vm );
   
   FALCON_FUNC  DBusPendingCall_wait( VMachine *vm );
   FALCON_FUNC  DBusPendingCall_completed( VMachine *vm );
   FALCON_FUNC  DBusPendingCall_cancel( VMachine *vm );
   
   FALCON_FUNC  DBusMessage_getDestination( VMachine *vm );
   FALCON_FUNC  DBusMessage_getSender( VMachine *vm );
   FALCON_FUNC  DBusMessage_getPath( VMachine *vm );
   FALCON_FUNC  DBusMessage_getInterface( VMachine *vm );
   FALCON_FUNC  DBusMessage_getMember( VMachine *vm );
   FALCON_FUNC  DBusMessage_getArgs( VMachine *vm );
   
   FALCON_FUNC DBusError_init( VMachine *vm );
}
}

#endif

/* end of dbus_ext.h */
