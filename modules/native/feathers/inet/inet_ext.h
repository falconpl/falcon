/*
   FALCON - The Falcon Programming Language.
   FILE: inet_ext.cpp

   Falcon VM interface to inet module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to socket module -- header.
*/


#ifndef _FALCON_INET_EXT_H_
#define _FALCON_INET_EXT_H_

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>

#include <falcon/classes/classshared.h>
#include <falcon/classes/classstream.h>

#ifndef FALCON_SOCKET_ERROR_BASE
   #define FALCON_SOCKET_ERROR_BASE        1170
#endif

#define FALSOCK_ERR_GENERIC  (FALCON_SOCKET_ERROR_BASE + 0)
#define FALSOCK_ERR_GENERIC_MSG "Generic network error"
#define FALSOCK_ERR_RESOLV  (FALCON_SOCKET_ERROR_BASE + 1)
#define FALSOCK_ERR_RESOLV_MSG "Network error during address resolution"
#define FALSOCK_ERR_CREATE  (FALCON_SOCKET_ERROR_BASE + 2)
#define FALSOCK_ERR_CREATE_MSG "Network error creating a socket"
#define FALSOCK_ERR_CONNECT  (FALCON_SOCKET_ERROR_BASE + 3)
#define FALSOCK_ERR_CONNECT_MSG "Network error during connection"
#define FALSOCK_ERR_SEND  (FALCON_SOCKET_ERROR_BASE + 4)
#define FALSOCK_ERR_SEND_MSG "Network error during send"
#define FALSOCK_ERR_RECV  (FALCON_SOCKET_ERROR_BASE + 5)
#define FALSOCK_ERR_RECV_MSG "Network error during receive"
#define FALSOCK_ERR_CLOSE  (FALCON_SOCKET_ERROR_BASE + 6)
#define FALSOCK_ERR_CLOSE_MSG "Network error during close"
#define FALSOCK_ERR_BIND  (FALCON_SOCKET_ERROR_BASE + 7)
#define FALSOCK_ERR_BIND_MSG "Network error during bind"
#define FALSOCK_ERR_ACCEPT  (FALCON_SOCKET_ERROR_BASE + 8)
#define FALSOCK_ERR_ACCEPT_MSG "Network error during accept"

#define FALSOCK_ERR_INCOMPATIBLE       (FALCON_SOCKET_ERROR_BASE + 9)
#define FALSOCK_ERR_INCOMPATIBLE_MSG   "Address incompatible for this kind of socket"

#define FALSOCK_ERR_UNRESOLVED         (FALCON_SOCKET_ERROR_BASE + 10)
#define FALSOCK_ERR_UNRESOLVED_MSG     "Unresolved address used in operation"

#define FALSOCK_ERR_FCNTL              (FALCON_SOCKET_ERROR_BASE + 11)
#define FALSOCK_ERR_FCNTL_MSG          "Error in FCNTL set/get"

#define FALSOCK_ERR_LISTEN             (FALCON_SOCKET_ERROR_BASE + 12)
#define FALSOCK_ERR_LISTEN_MSG         "Network error during listen"

#define FALSOCK_ERR_ADDRESS            (FALCON_SOCKET_ERROR_BASE + 13)
#define FALSOCK_ERR_ADDRESS_MSG        "Malformed network address"

#define FALSOCK_ERR_ALREADY_CREATED    (FALCON_SOCKET_ERROR_BASE + 14)
#define FALSOCK_ERR_ALREADY_CREATED_MSG "Already created"

#if WITH_OPENSSL
#define FALSOCK_ERR_SSLCONFIG          (FALCON_SOCKET_ERROR_BASE + 15)
#define FALSOCK_ERR_SSLCONFIG_MSG      "Error during SSL configuration"
#define FALSOCK_ERR_SSLCONNECT         (FALCON_SOCKET_ERROR_BASE + 16)
#define FALSOCK_ERR_SSLCONNECT_MSG     "Error during SSL negotiation"
#endif

namespace Falcon {
namespace Ext {

class ClassResolver: public ClassShared
{
public:
   ClassResolver();
   virtual ~ClassResolver();

   virtual int64 occupiedMemory( void* instance ) const;

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
};


class ClassAddress: public Class
{
public:
   ClassAddress();
   virtual ~ClassAddress();

   virtual int64 occupiedMemory( void* instance ) const;

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};


class ClassSocket: public Class
{
public:
   ClassSocket();
   virtual ~ClassSocket();

   virtual int64 occupiedMemory( void* instance ) const;

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual Selectable* getSelectableInterface( void* instance ) const;
};



class ClassSocketStream: public ClassStream
{
public:

   ClassSocketStream();
   virtual ~ClassSocketStream();
   virtual int64 occupiedMemory( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual Selectable* getSelectableInterface( void* instance ) const;
};


class ModuleInet: public Module
{
public:
   ModuleInet();
   virtual ~ModuleInet();

   Class* addressClass() const { return m_clsAddress; }
   Class* socketClass() const { return m_clsSocket; }
   Class* resolverClass() const { return m_clsResolver; }
   Class* socketStreamClass() const { return m_clsSocketStream; }

   #ifdef FALCON_SYSTEM_WIN
   Multiplex::Factory* selectMPXFactory() const { return m_smpxf; }
   #endif
public:
   Class* m_clsAddress;
   Class* m_clsSocket;
   Class* m_clsResolver;
   Class* m_clsSocketStream;

   #ifdef FALCON_SYSTEM_WIN
   Multiplex::Factory* m_smpxf;
   #endif
};


// =============================================
// Generic Functions
// ==============================================
FALCON_FUNC  falcon_getHostName( ::Falcon::VMachine *vm );
FALCON_FUNC  resolveAddress( ::Falcon::VMachine *vm );
FALCON_FUNC  socketErrorDesc( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_haveSSL( ::Falcon::VMachine *vm );

FALCON_DECLARE_ERROR( NetError );

}
}

#endif

/* end of socket_ext.h */
