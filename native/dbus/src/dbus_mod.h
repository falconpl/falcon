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
   Internal logic functions - declarations.
*/

#ifndef dbus_mod_H
#define dbus_mod_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>
#include <falcon/mt.h>
#include <falcon/vm.h>
#include <falcon/module.h>
#include <dbus/dbus.h>


namespace Falcon {
namespace Mod {

/**
   The inner wrapper.
   It can be "cloned", that is, just shared across cloned instances.
*/
class DBusWrapper: public FalconData
{
   class s_inner_data: public BaseAlloc
   {
   public:
      int32 m_refcount;
      DBusError m_err;
      DBusConnection* m_conn;
   };
   
   s_inner_data *m_content;
   
public:
   DBusWrapper();
   DBusWrapper( const DBusWrapper & );
   virtual ~DBusWrapper();
   
   inline DBusError *error() const { return &m_content->m_err; }
   inline DBusConnection *conn() const { return m_content->m_conn; }
   
   /** Connects to the DBUS.
      May return false on error -- check error().
    */
   bool connect();

   virtual FalconData* clone() const; // just increments the reference counter
   virtual void gcMark( uint32 mk );
};

/**
 * Error for all DBus errors.
 */
class f_DBusError: public ::Falcon::Error
{
public:
   f_DBusError():
      Error( "DBusError" )
   {}

   f_DBusError( const ErrorParam &params  ):
      Error( "DBusError", params )
      {}
};


/**
   Class used to manage the pending call items.
*/
/**
   The inner wrapper.
   It can be "cloned", that is, just shared across cloned instances.
*/
class DBusPendingWrapper: public FalconData
{
   DBusConnection* m_conn;
   DBusPendingCall* m_pc;
   
public:
   DBusPendingWrapper( DBusConnection*c, DBusPendingCall* p );
   DBusPendingWrapper( const DBusPendingWrapper & );
   virtual ~DBusPendingWrapper();
   
   inline DBusConnection *conn() const { return m_conn; }
   inline DBusPendingCall *pending() const { return m_pc; }

   virtual FalconData* clone() const; // just increments the reference counter
   virtual void gcMark( uint32 mk );
};


class DBusMessageWrapper: public FalconData
{
   DBusMessage* m_msg;
   
public:
   DBusMessageWrapper( DBusMessage *msg );
   DBusMessageWrapper( const DBusMessageWrapper & );
   virtual ~DBusMessageWrapper( void );
   
   inline DBusMessage *msg( void ) const { return m_msg; }
   
   
   virtual FalconData *clone( void ) const;
   virtual void gcMark( uint32 mk );
};


   
class FALCON_SERVICE DBusDispatcher: public Runnable
{
private:
   VMachine *m_vm;
   SysThread *m_th;
   DBusWrapper *m_wp;
   
   Event m_eTerminated;
public:
   DBusDispatcher( VMachine *vm, DBusWrapper *wp );
   virtual ~DBusDispatcher();
   
   virtual void *run( );
   
   /** Starts the parallel dispatcher. */
   void start();

   /** Stops the dispatcher

      Synchronously wait for clean termination and destroys this object.
      You can safely delete this object after this call.
   */
   void stop();
};

/** Initializes application wide data and clear up data at termination.

   This class is meant to prepare the application wide data at module initialization,
   and to shut down cleanly the dispatcher thread (if started) at application
   termination.
*/
class FALCON_SERVICE DBusModule: public Module
{
public:
   DBusModule();
   virtual ~DBusModule();
};


typedef struct _dbusHandlerData
{
   VMachine *vm;
   String *interface;
   String *name;
   CoreFunc *handler;
   bool isSignal;
} DBusHandlerData;


}
}

#endif

/* end of dbus_mod.h */
