/*
   The Falcon Programming Language
   FILE: dbus_dispatch_ext.cpp

   Falcon - DBUS official binding
   Interface extension functions
   -------------------------------------------------------------------
   Author: Enrico Lumetti
   Begin: Fri, 16 Apr 2010 19:43 +0100

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
   DBus parallel dispatcher
*/

#define FALCON_EXPORT_SERVICE
#include "dbus_mod.h"
#include "dbus_ext.h"

static Falcon::Mod::DBusDispatcher* s_dispatcher = 0;
static Falcon::Mutex* s_mtx_dispatcher = 0;

/*#
   @beginmodule dbus
*/

namespace Falcon {
namespace Ext {

/*#
   @method startDispath DBus

   Launches the message dispatcher for DBus receivers.
*/
FALCON_FUNC  DBus_startDispatch( VMachine *vm )
{
   s_mtx_dispatcher->lock();
   
   Mod::DBusWrapper* wp = static_cast< Mod::DBusWrapper* >( vm->self().asObject()->getUserData() );
   wp = static_cast< Mod::DBusWrapper* >( wp->clone() );
   
   if ( s_dispatcher != 0 )
   {
      s_dispatcher->stop();
      delete s_dispatcher;
   }
   s_dispatcher = new Mod::DBusDispatcher( vm, wp );
   s_dispatcher->start();
   s_mtx_dispatcher->unlock();
}

/*#
   @method stopDispatch DBus

   Stops the DBus message dispatcher.
*/

FALCON_FUNC  DBus_stopDispatch( VMachine *vm )
{
   s_mtx_dispatcher->lock();
   
   if ( s_dispatcher != 0 )
   {
      s_dispatcher->stop();
      delete s_dispatcher;
      s_dispatcher = 0;
   }
   s_mtx_dispatcher->unlock();
}

}

namespace Mod
{
   


FALCON_SERVICE DBusModule::DBusModule()
{
   s_mtx_dispatcher = new Mutex;
   s_dispatcher = 0; // to be sure
}


FALCON_SERVICE DBusModule::~DBusModule()
{
   DBusDispatcher *evt;
   s_mtx_dispatcher->lock();
   evt = s_dispatcher;
   s_dispatcher = 0;
   s_mtx_dispatcher->unlock();

   if( evt != 0 )
      evt->stop();
   
   delete s_dispatcher;
}

FALCON_SERVICE DBusDispatcher::DBusDispatcher( VMachine* vm, DBusWrapper *wp ):
   m_vm( vm ),
   m_th( 0 ),
   m_wp( wp )
{
   vm->incref();
}

FALCON_SERVICE DBusDispatcher::~DBusDispatcher()
{
   delete m_wp;
   m_vm->decref();
}

FALCON_SERVICE void* DBusDispatcher::run()
{
   while( ! m_eTerminated.wait( 20 ) )
   {
      dbus_connection_read_write_dispatch( m_wp->conn(), /*m_to*/ 10 );
   }

   return 0;
}

void FALCON_SERVICE DBusDispatcher::start()
{
   if ( m_th == 0 )
   {
      m_th = new SysThread( this );
      m_th->start();
   }
}

void FALCON_SERVICE DBusDispatcher::stop()
{
   if ( m_th != 0 )
   {
      m_eTerminated.set();
      void *dummy;
      m_th->join( dummy );
      m_th = 0;
   }
}

}
}
