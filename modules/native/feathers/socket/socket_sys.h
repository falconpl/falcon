/*
   FALCON - The Falcon Programming Language.
   FILE: socket_sys.h

   System dependant module specifications for socket module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependant module specifications for socket module
*/


#ifndef FLC_SOCKET_SYS_H
#define FLC_SOCKET_SYS_H

#include <falcon/string.h>
#include <falcon/falcondata.h>
#include <falcon/vm_sys.h>

#if WITH_OPENSSL
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#endif // WITH_OPENSSL

namespace Falcon {
namespace Sys {

//================================================
// Generic system dependant
//================================================

/** Initializes the system.
   Some incredibly idiotic OS (guess) needs to initialize the socket layer.
   On failure, the falcon socket module won't be initialized and the module
   creation function will return 0.
   \return true on success, false on failure.
*/
bool init_system();

#if WITH_OPENSSL
/** Initialize SSL subsystem.
 */
void ssl_init();

/** Terminate SSL subsystem (used when?).
 */
void ssl_fini();
#endif // WITH_OPENSSL

/** Retreives the host name for this machine */
bool getHostName( String &name );

bool getErrorDesc( int64 errorCode, String &desc );
bool getErrorDesc_GAI( int64 errorCode, String &desc );

//================================================
// Address
//================================================
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
      m_port(0),
      m_resolvCount(0),
      m_activeHostId(-1)
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

//================================================
// Socket
//================================================
/** Base class for system dependant socket implementation. */
class Socket: public Falcon::FalconData
{
protected:

   Address m_address;
   union {
      void *m_systemData;
      int m_iSystemData;
   } d;
   bool m_ipv6;
   int64 m_lastError;
   int32 m_timeout;
   int32 m_boundFamily;
   volatile int32 *m_refcount;


   Socket():
      m_lastError(0),
      m_timeout(0),
      m_boundFamily(0),
      m_refcount(0)
   {
      d.m_systemData = 0;
      m_refcount = (volatile int32*) memAlloc( sizeof(int32) );
      *m_refcount = 1;
   }

   Socket( void *systemData, bool ipv6 = false ):
      m_ipv6(ipv6 ),
      m_lastError(0),
      m_timeout(0),
      m_boundFamily(0),
      m_refcount(0)
   {
      d.m_systemData = systemData;
      m_refcount = (volatile int32*) memAlloc( sizeof(int32) );
      *m_refcount = 1;
   }

   Socket( const Socket& other );

public:
   virtual ~Socket();

   int64 lastError() const { return m_lastError; }
   int32 timeout() const { return m_timeout; }
   void timeout( int32 t ) { m_timeout = t; }

   int readAvailable( int32 msec,const Sys::SystemData *sysData = 0 );
   int writeAvailable( int32 msec, const Sys::SystemData *sysData = 0 );

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

   virtual void gcMark( uint32 mk ) {};
   virtual FalconData *clone() const;
};

//===============================================
// UDP Socket
//===============================================
class UDPSocket: public Socket
{

public:
   UDPSocket(Address &addr, bool ipv6 = false );
   UDPSocket( bool ipv6 = false );
   void turnBroadcast( bool mode );

   int32 recvFrom( byte *buffer, int32 size, Address &data );
   int32 sendTo( byte *buffer, int32 size, Address &data );
};


//================================================
// SSLData
//================================================
#if WITH_OPENSSL
/** Class bearing all SSL related data.
 */
class SSLData
{
public:

   /** Socket behavior.
    *  True if socket behaves as a server-side socket, false if as client.
    */
   bool asServer;

   /** Misc SSL errors codes.
    */
   typedef enum
   {
      no_error = 0,
      // happen in TCPSocket::sslconfig() :
      notready_error, /// Socket or SSL context not ready.
      ctx_error,      /// Error creating ssl context.
      cert_error,     /// Error setting certificate.
      pkey_error,     /// Error setting private key.
      ca_error,       /// Error setting cert authorities.
      handle_error,   /// Error creating ssl handle.
      fd_error,       /// Error attaching fd.
      // happen in TCPSocket::sslconnect(), sslwrite() or sslread() :
      already_error,       /// Connection attempt already done.
      notconnected_error,  /// Socket is not connected.
      handshake_failed,    /// Handshake has failed.
      write_error,         /// Error during write operation.
      read_error           /// Error during read operation.

   } ssl_error_t;

   /** Last SSL related error.
    */
   ssl_error_t lastSslError;

   /** Last SSL subsystem error.
    */
   int64 lastSysError;

   /** SSL protocols/versions.
    */
   typedef enum
   {
#ifndef OPENSSL_NO_SSL2
      SSLv2,
#endif
      SSLv3,
      SSLv23,
      TLSv1,
      DTLSv1

   } sslVersion_t;

   /** SSL version/method chosen (default SSLv3).
    *  Change it before creation of context.
    */
   sslVersion_t sslVersion;

   /** Handshake status.
    */
   typedef enum
   {
      handshake_todo,
      handshake_bad,
      handshake_ok

   } handshake_t;

   /** Wether handshake was done and how it resulted.
    */
   handshake_t handshakeState;

   SSL*         sslHandle;
   SSL_CTX*     sslContext;
   SSL_METHOD*  sslMethod;

   String   certFile; /// Certificate file.
   String   keyFile;  /// Private key file.
   String   caFile;   /// Certificate authorities file.

   SSLData()
      :
      asServer( false ),
      lastSslError( no_error ),
      lastSysError( 1 ),
      sslVersion( SSLv3 ),
      handshakeState( handshake_todo ),
      sslHandle( 0 ),
      sslContext( 0 ),
      sslMethod( 0 ),
      certFile( "" ),
      keyFile( "" ),
      caFile( "" )
   {}

   ~SSLData();

};
#endif // WITH_OPENSSL

//===============================================
// TCP Socket
//============================================
class TCPSocket: public Socket
{
   bool m_connected;
   #if WITH_OPENSSL
   SSLData* m_sslData;
   #endif

public:
   TCPSocket( bool ipv6 = false );
   TCPSocket( void *systemData, bool ipv6 = false):
      Socket( systemData, ipv6 ),
      m_connected( false )
      #if WITH_OPENSSL
      ,m_sslData( 0 )
      #endif
   {}

   TCPSocket( const TCPSocket& other );
   virtual ~TCPSocket();

   virtual FalconData* clone() const;
#if WITH_OPENSSL
   SSLData* ssl() const { return m_sslData; }
#endif

   /** Receive a buffer on the socket.
      The function waits m_timeout milliseconds (or forever if timeout is -1);
      if some data arrives in the meanwhile, or if the socket is closed,
      a read is performed. If m_timeout is 0, the function only reads available
      data.

      On timeout without data, the function returns -2; on error, it returns -1.
      If socket is closed on the other side, returns 0.
   */
   int32 recv( byte *buffer, int32 size );
   int32 send( const byte *buffer, int32 size );
   bool closeRead();
   bool closeWrite();
   bool close();
   bool isConnected();

   /** Connect to a server.
      Return false means error. Return true does not mean that
      connection is eshtablished. It is necessary to check for isConnected()
      to see if connection is ready. If timeout is zero, returns immediately,
      else returns when timeout expires or when connection is complete.
   */
   bool connect( Address &where );

#if WITH_OPENSSL
   /** Provide SSL capabilities to the socket.
    *  \param asServer True if we want server behavior, false for client behavior.
    *  \param sslVer Desired protocol
    *  \param certFile Certificate file (or empty string for none)
    *  \param pkeyFile Key file (or empty string for none)
    *  \return 0 on success or an error code.
    *  \note connect() must have been called before as we need a socket to attach.
    *
    *  Successful, that function leaves an SSLData instance as m_sslData.
    */
   SSLData::ssl_error_t sslConfig( bool asServer,
                                   SSLData::sslVersion_t sslVer,
                                   const char* certFile,
                                   const char* pkeyFile,
                                   const char* certAuthFile );

   /** Manually destroy the SSL context.
    *  Useful if you want to reuse/reconnect the socket.
    */
   void sslClear();

   /** Proceed with SSL handshake.
    */
   SSLData::ssl_error_t sslConnect();

   int32 sslWrite( const byte* buf, int32 sz );
   int32 sslRead( byte* buf, int32 sz );
#endif // WITH_OPENSSL
};

//================================================
// Server
//================================================
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
