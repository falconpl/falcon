/*
   FALCON - The Falcon Programming Language.
   FILE: socket_sys.h
   $Id: socket_sys.h,v 1.2 2007/06/23 10:14:51 jonnymind Exp $

   System dependant module specifications for socket module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   System dependant module specifications for socket module
*/


#ifndef FLC_SOCKET_SYS_H
#define FLC_SOCKET_SYS_H

#include <falcon/string.h>
#include <falcon/userdata.h>

namespace Falcon {
namespace Sys {

/** Initializes the system.
   Some incredibly idiotic OS (guess) needs to initialize the socket layer.
   On failure, the falcon socket module won't be initialized and the module
   creation function will return 0.
   \return true on success, false on failure.
*/
bool init_system();

/** Retreives the host name for this machine */
bool getHostName( String &name );

bool getErrorDesc( int64 errorCode, String &desc );

class Address
{
   void *m_systemData;

   String m_host;
   String m_service;
   int m_port;

   int32 m_resolvCount;
   int32 m_activeHostId;

   int64 m_lastError;

   friend class Socket;
   friend class TCPSocket;
   friend class UDPSocket;
   friend class ServerSocket;
public:

   Address():
      m_systemData(0),
      m_resolvCount(0),
      m_activeHostId(-1),
      m_port(0)
      {}

   ~Address();

   void set( const String &host )
   {
      m_host = host;
   }

   void set( const String &host, const String &service )
   {
      m_host = host;
      m_service = service;
   }


   bool getAddress( String &address ) const
   {
      if ( m_host.size() ) {
         address = m_host;
         return true;
      }
      return false;
   }

   bool getService( String &service ) const
   {
      if ( m_service.size() ) {
         service = m_service;
         return true;
      }
      return false;
   }

   int32 getPort() const { return m_port; }

   bool resolve();

   int32 getResolvedCount() const { return m_resolvCount; }

   /** Return resolved data.
      Use this to get addresses from autonomously created addresses, i.e.
      after a recvFrom or in sockets returned from accept().
      If returning false and last error is zero, then the function has been
      called incorrectly (i.e. no resolv performed); if lastError() returns
      a nonzero value, then a system error has occoured while retreiving
      canonical names.

      \param entry on success, will receive the canonical name of the required entry
      \param service on success, will receive the service name associate with the port number
      \param port on success, the numeric port value.
      \return true on success
   */
   bool getResolvedEntry( int32 count, String &entry, String &service, int32 &port );

   int32 activeHostID() const { return m_activeHostId; }

   /** Sets the entry that connect should use.
      Use this for repeated connection attemps to different IPs.
   */
   void activeHostID( int32 id ) { m_activeHostId = id; }

   /** Used internally by system module drivers.
      Connect() should use the data returned by this entry, and not transform
      autonomously the data in the host/service/port stored in the address;

      System specific methods will know how to handle the return value; i.e. on
      bsd like system, it will be a sockaddr structure to be casted to a
      sockaddr_in or sockaddr_in6.
   */
   void *getHostSystemData( int id ) const;

   int64 lastError() const { return m_lastError; }
};

/** Base class for system dependant socket implementation. */
class Socket: public Falcon::UserData
{
protected:

   Address m_address;
   void *m_systemData;
   bool m_ipv6;
   int64 m_lastError;
   int32 m_timeout;
   int32 m_boundFamily;


   Socket():
      m_lastError(0),
      m_timeout(0),
      m_boundFamily(0),
      m_systemData(0)
   {}

   Socket( void *systemData, bool ipv6 = false ):
      m_lastError(0),
      m_timeout(0),
      m_boundFamily(0),
      m_ipv6(ipv6 ),
      m_systemData( systemData )
   {}

public:
   ~Socket();

   int64 lastError() const { return m_lastError; }
   int32 timeout() const { return m_timeout; }
   void timeout( int32 t ) { m_timeout = t; }

   bool readAvailable( int32 msec );
   bool writeAvailable( int32 msec );

   /** Bind creates also the low-level socket.
      So we have to tell it if the socket to be created must be stream
      or packet, and if it's packet, if it has to support broadcasting.
   */
   bool bind( Address &addr, bool packet=false, bool broadcast=false );
   const Address &address() const { return m_address; }
   Address &address() { return m_address; }

   /** Ungraceful terminate.
      Clears the socket internal data without disconnecting.
   */
   void terminate();
};

class UDPSocket: public Socket
{

public:
   UDPSocket(Address &addr, bool ipv6 = false );
   UDPSocket( bool ipv6 = false );
   void turnBroadcast( bool mode );

   int32 recvFrom( byte *buffer, int32 size, Address &data );
   int32 sendTo( byte *buffer, int32 size, Address &data );
};

class TCPSocket: public Socket
{
   bool m_connected;

public:
   TCPSocket( bool ipv6 = false );
   TCPSocket( void *systemData, bool ipv6 = false):
      Socket( systemData, ipv6 )
   {}

   ~TCPSocket();

   /** Receive a buffer on the socket.
      The function waits m_timeout milliseconds (or forever if timeout is -1);
      if some data arrives in the meanwhile, or if the socket is closed,
      a read is performed. If m_timeout is 0, the function only reads available
      data.

      On timeout without data, the function returns -2; on error, it returns -1.
      If socket is closed on the other side, returns 0.
   */
   int32 recv( byte *buffer, int32 size );
   int32 send( byte *buffer, int32 size );
   bool closeRead();
   bool closeWrite();
   bool close();
   bool isConnected();

   /** Connect to a server.
      Return false means error. Return true does not mean that
      connection is eshtablished. It is necessary to check for isConnected()
      to see if connection is ready. If timeout is zero, returns immediately,
      else returns whe timeout expires or when connection is complete.
   */
   bool connect( Address &where );
};


class ServerSocket: public Socket
{
   bool m_bListening;

public:
   ServerSocket( bool ipv6 = true );
   ~ServerSocket();

   /** Accepts incoming calls.
      Returns a TCP socket on success, null if no new incoming data is arriving.
   */
   TCPSocket *accept();
};

}
}

#endif

/* end of socket_sys.h */
