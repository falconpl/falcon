/*
   FALCON - The Falcon Programming Language.
   FILE: inet_mod.h

   Module specifications for inet module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependant module specifications for socket module
*/


#ifndef _FALCON_FEATHERS_SOCKET_MOD_H_
#define _FALCON_FEATHERS_SOCKET_MOD_H_

#include <falcon/setup.h>

#ifdef FALCON_SYSTEM_WIN

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#define FALCON_SOCKET SOCKET
#define FALCON_SOCKLEN_T int
#define FALCON_EWOULDBLOCK WSAEWOULDBLOCK
#define FALCON_SOCKLEN_AS_INT socklen_t
#define FALCON_INVALID_SOCKET_VALUE INVALID_SOCKET

#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#define FALCON_SOCKET int
#define FALCON_SOCKLEN_T socklen_t
#define FALCON_SOCKLEN_AS_INT unsigned int
#define FALCON_EWOULDBLOCK EWOULDBLOCK
#define FALCON_INVALID_SOCKET_VALUE -1

#endif

#if WITH_OPENSSL
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif // WITH_OPENSSL


#include <falcon/string.h>
#include <falcon/refcounter.h>
#include <falcon/mt.h>
#include <falcon/shared.h>
#include <falcon/multiplex.h>
#include <falcon/selectable.h>
#include <falcon/stream.h>

namespace Falcon {
namespace Mod {

#ifndef socklen_t
#define socklen_t int
#endif

/** Initializes the system.
   Some incredibly idiotic OS (guess) needs to initialize the socket layer.
   On failure, the falcon socket module won't be initialized and the module
   creation function will return 0.
   \return true on success, false on failure.
*/
bool init_system();
bool shutdown_system();

#if WITH_OPENSSL
/** Initialize SSL subsystem.
 */
void ssl_init();

/** Terminate SSL subsystem (used when?).
 */
void ssl_fini();
#endif // WITH_OPENSSL

/** Retrieves the host name for this machine */
bool getHostName( String &name );

bool getErrorDesc( int64 errorCode, String &desc );

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

   uint32 m_lastError;


   uint32 m_mark;

public:

   Address();
   Address( const String& addr );
   Address( const String& host, const String& service );
   Address( const Address& other );

   static void convertToIPv6( void *vai, void* vsock6, socklen_t& sock6len );

   size_t occupiedMemory() const;
   void clear();

   void set( const String &host, const String &service );

   void setHost( const String &host )
   {
      clear();
      m_host = host;
   }

   bool getHost( String &address ) const
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

   bool parse( const String& addr );

   /** Synchronously resolve an address.
    @param useNameResolver if true, performs a synchronous query to the DNS system (if necessary).

    This operation fills the address with the system data valid to connect/bind the
    address given as textual representation.

    If the textual representation of the host is a numeric IPv4/IPv6 address,
    the call is non-blocking, otherwise, the system name resolver will be invoked.

    To prevent the name resolver to be invoked, pass useNameResolver = false. This
    will just translate an IP address representation into a system address.
    */
   bool resolve( bool useNameResolver = true );

   /** Returns true if the address has been resolved.
    *
    */
   bool isResolved() const;

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
   bool getResolvedEntry( int32 count, String &entry, String &service, int32 &port, int32& family );

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
   void* getRandomHostSystemData( int32 family ) const;


   uint32 lastError() const { return m_lastError; }
   void lastError( uint32 err ) {m_lastError = err; }

   void gcMark( uint32 mark ) { m_mark = mark; }
   uint32 currentMark() const { return m_mark; }

   void errorDesc( String& target );

   void toString( String& target ) const;
   String toString() const
   {
      String temp; toString(temp); return temp;
   }

protected:

private:
   virtual ~Address();
   FALCON_REFERENCECOUNT_DECLARE_INCDEC_NOEXPORT( Address );
};

//================================================
// Resolver
//================================================

/**
 Asynchronously resolve a given address.
 */
class Resolver: public Shared
{
public:
   Resolver( ContextManager* mgr, const Class* handler );
   void startResolving( Address* address );

   Address* address() const { return m_address; }

protected:
   virtual ~Resolver();

private:

   Address* m_address;

   class TH: public Runnable
   {
   public:
      TH( Resolver* res );
      virtual ~TH();
      virtual void* run();
   private:
      Resolver* m_res;
   };

   SysThread* m_thread;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC_NOEXPORT( Resolver );
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


//================================================
// Socket
//================================================
/** Base class for system dependent socket implementation. */
class Socket
{

public:
   Socket();
   Socket( int family, int type, int protocol );
   Socket( FALCON_SOCKET socket, int family, int type, int protocol, Address* remote );

   void create( int family, int type, int protocol );

   int family() const{ return m_family; }
   int type() const { return m_type; }
   int protocol() const { return m_protocol; }

   /** Get the internal socket descriptor */
   FALCON_SOCKET descriptor() const { return m_skt; }

   Stream* makeStreamInterface();
   Stream* stream() const;

   void gcMark( uint32 mk ) { m_mark = mk; }
   uint32 currentMark() const { return m_mark; }

   bool broadcasting() const;
   void broadcasting( bool mode );

   void closeRead();
   void closeWrite();
   void close();

   /** Connect to a server.
      Return false means error. Return true does not mean that
      connection is eshtablished. It is necessary to check for isConnected()
      to see if connection is ready. If timeout is zero, returns immediately,
      else returns when timeout expires or when connection is complete.

      @note modal operation
   */
   void connect( Address *where, bool async );

   /** Bind creates also the low-level socket.
      So we have to tell it if the socket to be created must be stream
      or packet, and if it's packet, if it has to support broadcasting.

      @note modal operation
   */
   void bind( Address *addr );

   void listen( int32 backlog = -1 );

   /** Accepts incoming calls.
      Returns a TCP socket on success, null if no new incoming data is arriving.

      @note Can be performed on a server socket only.
   */
   Socket *accept();

   int32 recv( byte *buffer, int32 size, Address *source=0 );
   int32 send( const byte *buffer, int32 size, Address *target=0 );

   /** Throws net error on error. */
   bool getBoolOption( int level, int option ) const;
   /** Throws net error on error. */
   int getIntOption( int level, int option ) const;
   /** Throws net error on error. */
   void getStringOption( int level, int option, String& value ) const;

   /** Throws net error on error. */
   void getDataOption( int level, int option, void* data, size_t& data_len ) const;

   /** Throws net error on error. */
   void setBoolOption( int level, int option, bool value );
   /** Throws net error on error. */
   void setIntOption( int level, int option, int value );
   /** Throws net error on error. */
   void setStringOption( int level, int option, const String& value );

   void setDataOption( int level, int option, const void* data, size_t data_len );

   void setOption( int level, int option, const Item& value);
   void getOption( int level, int option, Item& value ) const;

   bool isConnected() const;

   void setNonBlocking( bool mode ) const;
   bool isNonBlocking() const;

   Address* address() const {return m_address;}

   /** True when a zero is read after a blocking operation.

   This is set to true when the remote side has signaled that the
   it won't send more data.

   The socket might be closed on the other side, or the process might
   have crashed.

   Conversely the socket might still be open to receive more data
   from this side. The only way to reliable know this information
   is trying to send data and detect any error.
   */
   bool eof() const { return m_bEof; }

   /**
    * Gets a system-resolved address that is suitable to be used by this socket.
    */
   int getValidAddress( Address* addr, struct sockaddr_storage& target, FALCON_SOCKLEN_T& tglen ) const;

   #if WITH_OPENSSL
      SSLData* ssl() const { return m_sslData; }
   #endif


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
   void sslConfig( bool asServer,
                                   SSLData::sslVersion_t sslVer,
                                   String* certFile,
                                   String* pkeyFile,
                                   String* certAuthFile );

   /** Manually destroy the SSL context.
    *  Useful if you want to reuse/reconnect the socket.
    */
   void sslClear();

   /** Proceed with SSL handshake.
    */
   void sslConnect();

   int32 sslWrite( const byte* buf, int32 sz );
   int32 sslRead( byte* buf, int32 sz );

   /* These are called internally by send/recv
    */
#endif // WITH_OPENSSL

protected:
   int m_family;
   int m_type;
   int m_protocol;

   Stream* m_stream;
   Address* m_address;
   mutable bool m_bConnected;
   FALCON_SOCKET m_skt;

   uint32 m_mark;
   bool m_bEof;
#ifdef FALCON_SYSTEM_WIN
   mutable bool m_bioMode;
#endif

   virtual ~Socket();

   typedef enum {
         param_bool,
         param_int,
         param_string,
         param_data,
         param_linger,
         param_unknown,
   }
   t_option_type;

   static t_option_type getOptionType( int level, int option );

private:
   #if WITH_OPENSSL
   SSLData* m_sslData;
   #endif

   int sys_getsockopt( int level, int option_name, void *option_value, FALCON_SOCKLEN_AS_INT * option_len) const;
   int sys_setsockopt( int level, int option_name, const void *option_value, FALCON_SOCKLEN_AS_INT option_len) const;

   // disallow evil constructor
   Socket( const Socket& ) {};

   FALCON_REFERENCECOUNT_DECLARE_INCDEC_NOEXPORT( Socket );
};


/**
 Selectable interface for the Socket class.
 */
class SocketSelectable: public FDSelectable
{
public:
   SocketSelectable( const Class* cls, Socket* inst );
   virtual ~SocketSelectable();
   virtual const Multiplex::Factory* factory() const;
   virtual int getFd() const;
};


//================================================
// Socket
//================================================

class SocketStream: public Stream
{
public:
   SocketStream( Socket* skt );
   virtual ~SocketStream();

   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );
   virtual bool close();
   virtual Stream *clone() const;

   virtual const Multiplex::Factory* multiplexFactory() const;

   virtual int64 tell();
   virtual bool truncate( off_t pos=-1 );
   virtual off_t seek( off_t pos, e_whence w );

   virtual const Class* handler() const;

   class Selectable: public FDSelectable
   {
   public:
      Selectable( SocketStream* inst );
      virtual ~Selectable();
      virtual const Multiplex::Factory* factory() const;
      virtual int getFd() const;
   };

   Socket* skt() const { return m_socket; }

   static void setHandler( Class* handler );
private:
   Socket* m_socket;

   static Class* m_socketStreamHandler;
};


#if WITH_OPENSSL
void ssl_init();
void ssl_fini();
#endif
}
}

#endif

/* end of inet_mod.h */
