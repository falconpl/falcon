/*
   FALCON - The Falcon Programming Language.
   FILE: socket_sys_win.cpp

   UNIX/BSD system specific interface to sockets.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   UNIX/BSD system specific interface to sockets.
*/

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <unistd.h>
#include <falcon/autocstring.h>
#include <falcon/vm_sys_posix.h>

#include <netdb.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include "socket_sys.h"

// Sun doesn't provide strerror_r
#if ( defined (__SUNPRO_CC) && __SUNPRO_CC <= 0x580 )
static int strerror_r(int errnum, char * buf, unsigned n)
{
   ::strncpy(buf, strerror(errnum), n);
   buf[n-1] = '\0'; // just in case...
   return 0;
}
#endif

namespace Falcon {
namespace Sys {

//================================================
// Generic system dependant
//================================================

bool init_system()
{
   return true;
}

#if WITH_OPENSSL
static bool bSslInitialized = false;

void ssl_init()
{
   if ( !bSslInitialized )
   {
      SSL_load_error_strings();
      SSL_library_init();
      bSslInitialized = true;
   }
}

void ssl_fini()
{
   if ( bSslInitialized )
   {
      ERR_free_strings();
      bSslInitialized = false;
   }
}
#endif // WITH_OPENSSL

bool isIPV4( const String &ipv4__ )
{
   String ipv4 = ipv4__;
   struct addrinfo hints;
   struct addrinfo *res = 0;

   // we want to see an IPv4:
   memset( &hints, 0, sizeof( hints ) );

   hints.ai_family = AF_INET;
   hints.ai_flags = AI_NUMERICHOST;

   //toCString is guaranteed to stay as long as the string exists.
   char addrBuff[256];
   ipv4.toCString( addrBuff, 255 );
   int error = getaddrinfo( addrBuff, 0, &hints, &res );
   if ( error == EAI_NONAME )
      return false;

   freeaddrinfo( res );

   return true;
}

bool isIPV6( const String &ipv6__ )
{
   String ipv6 = ipv6__;
   struct addrinfo hints;
   struct addrinfo *res = 0;

   // we want to see an IPv4:
   memset( &hints, 0, sizeof( hints ) );

   hints.ai_family = AF_INET6;
   hints.ai_flags = AI_NUMERICHOST;

   //toCString is guaranteed to stay as long as the string exists.
   char addrBuff[256];
   ipv6.toCString( addrBuff, 255 );
   int error = getaddrinfo( addrBuff, 0, &hints, &res );
   if ( error == EAI_NONAME )
      return false;

   freeaddrinfo( res );

   return true;
}


bool getHostName( String &name )
{
   char hostName[256];
   if ( ::gethostname( hostName, 255 ) == 0 ) {
      name.bufferize( hostName );
      return true;
   }

   return false;
}

bool getErrorDesc( int64 error, String &ret )
{
   char buf[512];
   if ( error ==  -1 )
      ret.bufferize( "(internal) No valid target addresses for selected protocol" );
   else if ( strerror_r( (int) error, buf, 511 ) != 0 )
      ret.bufferize( "(internal) Unknown error" );
   else
      ret.bufferize( buf );
   return true;
}

bool getErrorDesc_GAI( int64 error, String &ret )
{
   const char* err;
   if ( error ==  -1 )
      ret.bufferize( "(internal) No valid target addresses for selected protocol" );
   else if ( (err = gai_strerror( (int) error )) == 0 )
      ret.bufferize( "(internal) Unknown error" );
   else
      ret.bufferize( err );
   return true;
}




//================================================
// Address
//================================================

Address::~Address()
{
   if ( m_systemData ) {
      struct addrinfo *res = (struct addrinfo *) m_systemData;
      freeaddrinfo( res );
   }
}

bool Address::resolve()
{
   struct addrinfo hints;
   struct addrinfo *res = 0;

   // we want to see an IPv4:
   memset( &hints, 0, sizeof( hints ) );
   hints.ai_family = AF_UNSPEC;

   // check if the service is numeric
   int64 port = 0;
   m_service.parseInt( port );

   //toCString is guaranteed to stay as long as the string exists.
   AutoCString hostBuf( m_host );
   AutoCString srvBuf( m_service );
   int error = getaddrinfo( hostBuf.c_str(),
                            (port != 0 || m_service == "") ? 0 : srvBuf.c_str(),
                            &hints, &res ) ;
   if ( error != 0 ) {
      m_lastError = (int64) error;
      return false;
   }

   struct addrinfo *oldres = (struct addrinfo *) m_systemData;
   if ( oldres != 0)
      freeaddrinfo( oldres );
   m_systemData = res;

   m_resolvCount = 0;
   while ( res != 0 ) {
      m_resolvCount ++;
      // valid for both ipv4 and ipv6
      if( port != 0 )
      {
         ((struct sockaddr_in*) res->ai_addr)->sin_port = htons( (int) port);
      }
      res = res->ai_next;
   }

   return true;
}

bool Address::getResolvedEntry( int32 count, String &entry, String &service, int32 &port )
{
   m_lastError = 0;

   if ( m_systemData == 0 )
      return false;

   struct addrinfo *res = (struct addrinfo *) m_systemData;
   while( res != 0 && count > 0 ) {
      count --;
      res = res->ai_next;
   }

   if ( res == 0 )
      return false;

   char host[256];
   char serv[32];
   // try to resolve the service name
   int error = getnameinfo( res->ai_addr, res->ai_addrlen, host, 255, serv, 31, NI_NUMERICHOST );
   // try with the service number.
   if ( error != 0 )
      error = getnameinfo( res->ai_addr, res->ai_addrlen, host, 255, serv, 31, NI_NUMERICHOST | NI_NUMERICSERV );
   if ( error == 0 ) {
      entry.bufferize( host );
      service.bufferize( serv );
      // port is in the same position for ip and ipv6
      struct sockaddr_in *saddr = (struct sockaddr_in *)res->ai_addr;
      port = ntohs( saddr->sin_port );
      return true;
   }

   m_lastError = error;
   return false;
}

void *Address::getHostSystemData( int id ) const
{
   struct addrinfo *res = (struct addrinfo *) m_systemData;
   while( res != 0 && id > 0 ) {
      id --;
      res = res->ai_next;
   }
   return res;
}

//================================================
// Socket
//================================================

static int s_select( int skt, int32 msec, int32 mode )
{
   struct timeval tv, *tvp;
   fd_set set;

   FD_ZERO( &set );
   FD_SET( skt, &set );
   if ( msec >= 0 ) {
      tvp = &tv;
      tv.tv_sec = msec / 1000;
      tv.tv_usec = (msec % 1000 ) * 1000;
   }
   else
      tvp = 0;

   int count;

   switch( mode )
   {
      case 0: count = select( skt + 1, &set, 0, 0, tvp ); break;
      case 1: count = select( skt + 1, 0, &set, 0, tvp ); break;
      case 2: count = select( skt + 1, 0, 0, &set, tvp ); break;
   }

   return count;
}

static int s_select_connect( int skt, int32 msec )
{
   struct timeval tv, *tvp;
   fd_set write_set, error_set;

   FD_ZERO( &write_set );
   FD_SET( skt, &write_set );
   FD_ZERO( &error_set );
   FD_SET( skt, &error_set );

   if ( msec >= 0 ) {
      tvp = &tv;
      tv.tv_sec = msec / 1000;
      tv.tv_usec = (msec % 1000 ) * 1000;
   }
   else
      tvp = 0;

   int count = select( skt + 1, 0, &write_set, &error_set, tvp );
   // nothing done
   if ( count == 0 )
      return 0;

   if ( FD_ISSET( skt, &write_set ) )
      return 1; // connection succesful


   return -1; // error
}

/*
Socket::~Socket()
{
   // ungraceful close.
   terminate();
}
*/

void Socket::terminate()
{
   if ( d.m_iSystemData == 0 )
      return;

   int s = (int) d.m_iSystemData;
   ::close( s );
   d.m_iSystemData = 0;
}

int Socket::readAvailable( int32 msec, const Sys::SystemData *sysData )
{
   m_lastError = 0;
   struct timeval tv, *tvp;
   fd_set set;
   int last;

   FD_ZERO( &set );
   FD_SET( d.m_iSystemData, &set );
   if( sysData != 0 )
   {
      last = sysData->m_sysData->interruptPipe[0];
      FD_SET( last, &set );
      if( last < d.m_iSystemData )
         last = d.m_iSystemData;
   }
   else
      last = d.m_iSystemData;

   if ( msec >= 0 ) {
      tvp = &tv;
      tv.tv_sec = msec / 1000;
      tv.tv_usec = (msec % 1000 ) * 1000;
   }
   else
      tvp = 0;

   switch( select( last + 1, &set, 0, 0, tvp ) )
   {
      case 1:
      case 2:
         if ( sysData != 0 && FD_ISSET( sysData->m_sysData->interruptPipe[0], &set ) )
         {
            return -2;
         }

      return 1;

      case -1:
         if( errno == EINPROGRESS ) {
            m_lastError = 0;
            return 0;
         }
         m_lastError = errno;
      return -1;
   }

   return 0;
}

int Socket::writeAvailable( int32 msec, const Sys::SystemData *sysData )
{
   m_lastError = 0;
   struct pollfd poller[2];
   int fds;

   poller[0].fd = (int) d.m_iSystemData;
   poller[0].events = POLLOUT;

   if ( sysData != 0 )
   {
      fds = 2;
      poller[1].fd = sysData->m_sysData->interruptPipe[0];
      poller[1].events = POLLIN;
   }
   else
      fds = 1;

   int res;
   while( ( res = poll( poller, fds, msec ) ) == EAGAIN );

   if ( res > 0 )
   {
      if( sysData != 0 && (poller[1].revents & POLLIN) != 0 )
      {
         return -2;
      }

      if( (poller[0].revents & ( POLLOUT | POLLHUP ) ) != 0 )
         return 1;
   }
   else {
      m_lastError = errno;
      return -1;
   }

   return 0;
}

bool Socket::bind( Address &addr, bool packet, bool broadcast )
{
   // has the address to be resolved?
   if( addr.getResolvedCount() == 0 ) {
      if ( ! addr.resolve() ) {
         m_lastError = addr.m_lastError;
         return false;
      }
   }

   // try to bind to the resovled host
   struct addrinfo *ai = 0;
   int skt = -1;
   // find a suitable address.
   int entryId;
   int type = packet ? SOCK_DGRAM: SOCK_STREAM;

   for ( entryId = 0; entryId < addr.getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)addr.getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == AF_INET ) {
         skt = socket( ai->ai_family, type, ai->ai_protocol );
         if ( skt > 0 ) {
            break;
         }
      }
   }

   if ( skt == -1 ) {
      m_lastError = -1;
      return false;
   }

   // dispose of old socket
   if ( d.m_iSystemData != 0 ) {
      ::close( (int) d.m_iSystemData );
      d.m_iSystemData = 0;
   }

   // need of a broadcast semantic?
   if ( broadcast )
   {
      int iOpt = 1;
      setsockopt( skt, SOL_SOCKET, SO_BROADCAST, (const char *) &iOpt, sizeof( iOpt ));
   }

   // always useful
   {
      int iOpt = 1;
      setsockopt( skt, SOL_SOCKET, SO_REUSEADDR, (const char *) &iOpt, sizeof( iOpt ));
   }

   int res = ::bind( skt, ai->ai_addr, ai->ai_addrlen );
   d.m_iSystemData = skt;

   // success!!!
   if ( res == 0 ) {
      m_boundFamily = ai->ai_family;
      addr.getResolvedEntry( entryId, m_address.m_host, m_address.m_service, m_address.m_port );
      // we'll resolve the entry again if the port is needed later on.
      return true;
   }

   m_lastError = errno;
   return false;
}

//================================================
// SSLData
//================================================
#if WITH_OPENSSL

SSLData::~SSLData()
{
   if ( sslContext )
   {
      SSL_CTX_free( sslContext );
      sslContext = 0;
   }
   if ( sslHandle )
   {
      SSL_shutdown( sslHandle );
      SSL_free( sslHandle );
      sslHandle = 0;
   }
}

#endif // WITH_OPENSSL

//===============================================
// TCP Socket
//===============================================

TCPSocket::TCPSocket( bool ipv6 )
{
   m_ipv6 = ipv6;
   m_connected = false;
   // default timeout is zero
   m_timeout = 0;
   m_lastError = 0;
   #if WITH_OPENSSL
   m_sslData = 0;
   #endif
   d.m_iSystemData = 0;
}

TCPSocket::~TCPSocket()
{
   #if WITH_OPENSSL
   if ( m_sslData )
   {
      delete m_sslData;
      m_sslData = 0;
   }
   #endif
}

int32 TCPSocket::recv( byte *buffer, int32 size )
{
#if WITH_OPENSSL
   // transfer to sslRead ?
   if ( m_sslData && m_sslData->handshakeState == SSLData::handshake_ok )
      return sslRead( buffer, size );
#endif

   if ( ! readAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int read = ::recv( (int) d.m_iSystemData, (char *)buffer, size, 0 );
   if ( read < 0 ) {
      m_lastError = errno;
      return -1;
   }
   if ( read == 0 ) // shutdown detected
      terminate();

   return read;
}

int32 TCPSocket::send( const byte *buffer, int32 size )
{
#if WITH_OPENSSL
   // transfer to sslWrite?
   if ( m_sslData && m_sslData->handshakeState == SSLData::handshake_ok )
      return sslWrite( buffer, size );
#endif

   if ( ! writeAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int sent = ::send( (int) d.m_iSystemData, (char *) buffer, size, 0 );
   if ( sent < 0 ) {
      m_lastError = errno;
      return -1;
   }

   return sent;
}

bool TCPSocket::closeRead()
{
   if ( shutdown( (int) d.m_iSystemData, SHUT_RD ) < 0 )
   {
      m_lastError = errno;
      return false;
   }

   // wait to receive notify
   if ( m_timeout != 0 ) {
      readAvailable( m_timeout );
   }
   return true;
}

bool TCPSocket::closeWrite()
{
   if ( shutdown( (int) d.m_iSystemData, SHUT_WR ) < 0 ) {
      m_lastError = errno;
      return false;
   }
   // no wait
   return true;
}


bool TCPSocket::close()
{
   if ( shutdown( (int) d.m_iSystemData, SHUT_RDWR ) < 0 )
   {
      m_lastError = errno;
      return false;
   }

   // wait to receive notify
   if ( m_timeout != 0 ) {
      readAvailable( m_timeout );
   }
   return true;
}

bool TCPSocket::connect( Address &where )
{
   m_lastError = 0;
   int flags = 0;

   // let's try to connect the addresses in where.
   if ( where.getResolvedCount() == 0 ) {
      if ( ! where.resolve() ) {
         m_lastError = where.m_lastError;
         return false;
      }
   }

   struct addrinfo *ai = 0;
   int skt = -1;
   // find a suitable address.
   int entryId;
   for ( entryId = 0; entryId < where.getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)where.getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == AF_INET ) {
         skt = socket( ai->ai_family, SOCK_STREAM, ai->ai_protocol );
         if ( skt > 0 ) {
            break;
         }
      }
   }

   // quite impossible but...
   if ( skt == -1 ) {
      m_lastError = -1;
      return false;
   }

   // dispose of old socket
   if ( d.m_iSystemData != 0 ) {
      ::close( (int) d.m_iSystemData );
      d.m_iSystemData = 0;
   }

   int bOptVal = 1;

   // set keepalive
   if ( setsockopt( skt, SOL_SOCKET, SO_KEEPALIVE, (char *) &bOptVal, sizeof( int ) ) < 0 )
   {
      m_lastError = errno;
      return false;
   }

   // if timeout is -1, do not set nonblocking.
   if( m_timeout >= 0 ) {
      // set nonblocking
      flags = fcntl( skt, F_GETFL );
      flags |= O_NONBLOCK;
      fcntl( skt, F_SETFL, flags );
   }

   d.m_iSystemData = skt;
   m_lastError = 0;
   m_connected = false;
   int res = ::connect( skt, ai->ai_addr, ai->ai_addrlen );

   // reset nonblocking status
   if ( m_timeout >= 0 ) {
      flags &= ~O_NONBLOCK;
      fcntl( skt, F_SETFL, flags );
   }

   where.getResolvedEntry( entryId, m_address.m_host, m_address.m_service, m_address.m_port );

   if ( res < 0 ) {
      m_lastError = errno;

      if( m_lastError == EINPROGRESS )
         m_lastError = 0;
      else {
         return false;
      }
   }
   else {
      m_connected = true;
      return true;
   }

   // we are not connected; do we have to wait?

   // select/wait?
   if ( m_timeout > 0 ) {
      res = s_select_connect( skt, m_timeout );
      if ( res == 1 ) {
         m_connected = true;
         return true;
      }
      else if ( res == -1 ) {
         // An error
         m_lastError = errno;
      }
      else
         m_lastError = 0;
      // else still nothing
   }

   // not connected.
   return false;
}

bool TCPSocket::isConnected()
{
   if ( m_connected )
   {
      return true;
   }

   int status = s_select_connect( (int) d.m_iSystemData, m_timeout );
   if ( status == 1 ) {
      m_connected = true;
      return true;
   }
   else if( status == -1 )
      m_lastError = errno;
   else
      m_lastError = 0;

   return false;
}

#if WITH_OPENSSL
SSLData::ssl_error_t TCPSocket::sslConfig( bool asServer,
                                           SSLData::sslVersion_t sslVer,
                                           const char* certFile,
                                           const char* pkeyFile,
                                           const char* certAuthFile )
{
   Falcon::Sys::ssl_init();

   if ( d.m_iSystemData <= 0 ) // no socket to attach
      return SSLData::notready_error;

   if ( m_sslData ) // called before?
      return SSLData::no_error;

   int i;
   SSLData* sslD = new SSLData;
   sslD->asServer = asServer;

   // choose method
   sslD->sslVersion = sslVer;
   switch ( sslVer )
   {
#ifndef OPENSSL_NO_SSL2
   //case SSLData::SSLv2:  sslD->sslMethod = (SSL_METHOD*) SSLv2_method(); break;
#endif
   //case SSLData::SSLv3:  sslD->sslMethod = (SSL_METHOD*) SSLv3_method(); break;
   case SSLData::SSLv23: sslD->sslMethod = (SSL_METHOD*) SSLv23_method(); break;
   case SSLData::TLSv1:  sslD->sslMethod = (SSL_METHOD*) TLSv1_method(); break;
   case SSLData::DTLSv1: sslD->sslMethod = (SSL_METHOD*) DTLSv1_method(); break;
   default: sslD->sslMethod = (SSL_METHOD*) SSLv23_method();
   }

   // create context
   sslD->sslContext = SSL_CTX_new( sslD->sslMethod );
   if ( !sslD->sslContext )
   {
      delete sslD;
      return SSLData::ctx_error;
   }

   // certificate file
   if ( certFile && certFile[0] != '\0' )
   {
      if ( ( i = SSL_CTX_use_certificate_file( sslD->sslContext, certFile,
          SSL_FILETYPE_PEM ) != 1 ) )
      {
         delete sslD;
         m_lastError = i;
         return SSLData::cert_error;
      }
      sslD->certFile = certFile;
      sslD->certFile.bufferize();
   }

   // private key file
   if ( pkeyFile && pkeyFile[0] != '\0' )
   {
      if ( ( i = SSL_CTX_use_PrivateKey_file( sslD->sslContext, pkeyFile,
          SSL_FILETYPE_PEM ) != 1 ) )
      {
         delete sslD;
         m_lastError = i;
         return SSLData::pkey_error;
      }
      sslD->keyFile = pkeyFile;
      sslD->keyFile.bufferize();
   }

   // certificates authorities
   if ( certAuthFile && certAuthFile[0] != '\0' )
   {
      STACK_OF( X509_NAME ) *cert_names;
      cert_names = SSL_load_client_CA_file( certAuthFile );
      if ( cert_names != 0 )
      {
         SSL_CTX_set_client_CA_list( sslD->sslContext, cert_names );
      }
      else
      {
         delete sslD;
         m_lastError = i;
         return SSLData::ca_error;
      }
      sslD->caFile = certAuthFile;
      sslD->caFile.bufferize();
   }

   // ssl handle
   sslD->sslHandle = SSL_new( sslD->sslContext );
   if ( !sslD->sslHandle )
   {
      delete sslD;
      return SSLData::handle_error;
   }

   // attach file descriptor
   if ( ( i = SSL_set_fd( sslD->sslHandle, d.m_iSystemData ) ) != 1 )
   {
      delete sslD;
      m_lastError = i;
      return SSLData::fd_error;
   }

   // done
   m_sslData = sslD;
   return SSLData::no_error;
}

void TCPSocket::sslClear()
{
   if ( m_sslData )
   {
      delete m_sslData;
      m_sslData = 0;
   }
}

SSLData::ssl_error_t TCPSocket::sslConnect()
{
   //int flags = 0;
   int i;

   // need ssl context
   if ( !m_sslData )
      return SSLData::notready_error;
   // no need to call several times
   if ( m_sslData->handshakeState != SSLData::handshake_todo )
      return SSLData::already_error;
   // socket needs to be connected
   if ( !m_connected )
      return SSLData::notconnected_error;
#if 0
   // if timeout is -1, do not set nonblocking.
   if( m_timeout >= 0 )
   {
      // set nonblocking
      flags = fcntl( d.m_iSystemData, F_GETFL );
      flags |= O_NONBLOCK;
      fcntl( d.m_iSystemData, F_SETFL, flags );
   }
#endif

   if ( m_sslData->asServer ) // server-side socket
   {
      i = SSL_accept( m_sslData->sslHandle );
   }
   else // client-side socket
   {
      i = SSL_connect( m_sslData->sslHandle );
   }
#if 0
   // reset nonblocking status
   if ( m_timeout >= 0 )
   {
      flags &= ~O_NONBLOCK;
      fcntl( d.m_iSystemData, F_SETFL, flags );
   }
#endif
   if ( i != 1 )
   {
      m_sslData->lastSysError = SSL_get_error( m_sslData->sslHandle, i );
      m_sslData->lastSslError = SSLData::handshake_failed;
      m_lastError = m_sslData->lastSysError;
      m_sslData->handshakeState = SSLData::handshake_bad;
      return SSLData::handshake_failed;
   }

   m_sslData->handshakeState = SSLData::handshake_ok;
   return SSLData::no_error;
}

int32 TCPSocket::sslWrite( const byte* buf, int32 sz )
{
   int i = SSL_write( m_sslData->sslHandle, buf, sz );
   if ( i <= 0 )
   {
      m_sslData->lastSysError = SSL_get_error( m_sslData->sslHandle, i );
      m_sslData->lastSslError = SSLData::write_error;
      m_lastError = m_sslData->lastSysError;
      return -1;
   }
   return i;
}

int32 TCPSocket::sslRead( byte* buf, int32 sz )
{
   int i = SSL_read( m_sslData->sslHandle, buf, sz );
   if ( i <= 0 )
   {
      m_sslData->lastSysError = SSL_get_error( m_sslData->sslHandle, i );
      m_sslData->lastSslError = SSLData::read_error;
      m_lastError = m_sslData->lastSysError;
      return -1;
   }
   return i;
}
#endif // WITH_OPENSSL

//================================================
// Server
//================================================

ServerSocket::ServerSocket( bool ipv6 ):
   Socket( 0, ipv6 ),
   m_bListening( false )
{
}

ServerSocket::~ServerSocket()
{
}

TCPSocket *ServerSocket::accept()
{
   int srv = (int) d.m_iSystemData;

   if ( ! m_bListening ) {
      if ( ::listen( srv, SOMAXCONN ) != 0 ) {
         m_lastError = errno;
         return 0;
      }
      m_bListening = true;
   }

   if ( s_select( srv, m_timeout, 0 ) ) {
      socklen_t addrlen;
      struct sockaddr *address;
      struct sockaddr_in6 addrIn6;
      struct sockaddr_in addrIn;
      // where is this socket bound?
      if( m_boundFamily == AF_INET ) {
         address = (struct sockaddr *) &addrIn;
         addrlen = sizeof( addrIn );
      }
      else {
         address = (struct sockaddr *) &addrIn6;
         addrlen = sizeof( addrIn6 );
      }

      int skt = ::accept( srv, address, &addrlen );
      TCPSocket *s = new TCPSocket( (void *) &skt );

      char hostName[64];
      char servName[64];

      if ( getnameinfo( address, addrlen, hostName, 63, servName, 63, NI_NUMERICHOST | NI_NUMERICSERV ) == 0 )
      {
         String host, serv;
         host.bufferize( hostName );
         serv.bufferize( servName );
         s->address().set( host, serv );
      }

      return s;
   }

   return 0;
}

//===============================================
// UDP Socket
//===============================================

UDPSocket::UDPSocket( Address &addr, bool ipv6 ):
   Socket( 0, ipv6 )
{
   this->bind( addr, true );
}

UDPSocket::UDPSocket( bool ipv6 ):
   Socket( 0, ipv6 )
{
   // creating an unbound socket
   int skt = socket( ipv6, SOCK_DGRAM, 0 );
   if ( skt == -1 ) {
      m_lastError = errno;
   }
   else {
      d.m_iSystemData = skt;
   }
}

void UDPSocket::turnBroadcast( bool mode )
{
   int iOpt = mode ? 1:0;
   int skt = (int) d.m_iSystemData;
   setsockopt( skt, SOL_SOCKET, SO_BROADCAST, (const char *) &iOpt, sizeof( iOpt ));
}

int32 UDPSocket::recvFrom( byte *buffer, int32 size, Address &data )
{
   int s = (int) d.m_iSystemData;
   // sockaddr_in6 should be the longest possible structure we may receive.
   struct sockaddr_in6 addr;
   struct sockaddr *paddr = (struct sockaddr *) &addr;
   socklen_t len = sizeof( addr );

   if ( ! readAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int32 retsize = ::recvfrom( s, buffer, size, 0, paddr, &len );

   if ( retsize == -1 )
      m_lastError = errno;
   else {
      // save address
      char host[64];
      char serv[31];
      int error = getnameinfo( paddr, len, host, 63, serv, 31, NI_NUMERICHOST | NI_NUMERICSERV );
      if ( error == 0 ) {
         String shost;
         shost.bufferize( host );
         String sserv;
         sserv.bufferize( serv );
         data.set( shost, sserv );
      }
      else {
         m_lastError = errno;
         return -1;
      }

      m_lastError = 0;
   }

   return retsize;
}

int32 UDPSocket::sendTo( byte *buffer, int32 size, Address &where )
{
   int s = (int) d.m_iSystemData;

   // let's try to connect the addresses in where.
   if ( where.getResolvedCount() == 0 ) {
      if ( ! where.resolve() ) {
         m_lastError = where.m_lastError;
         return false;
      }
   }

   struct addrinfo *ai = 0;
   // find a suitable address.
   int entryId;
   for ( entryId = 0; entryId < where.getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)where.getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == AF_INET ) {
         break;
      }
   }

   // quite impossible but...
   if ( entryId == where.getResolvedCount() ) {
      m_lastError = -1;
      return false;
   }


   if ( ! writeAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int32 retsize = ::sendto( s, buffer, size, 0, ai->ai_addr, ai->ai_addrlen );

   if ( retsize == -1 )
      m_lastError = errno;
   else {
      m_lastError = 0;
   }

   return retsize;
}

} // namespace
}

/* end of socket_sys_win.cpp */
