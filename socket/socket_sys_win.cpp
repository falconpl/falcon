/*
   FALCON - The Falcon Programming Language.
   FILE: socket_sys_win.cpp

   MS-Windows system specific interface to sockets.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   MS-Windows system specific interface to sockets.
*/

#include <winsock2.h>
#include <ws2tcpip.h>

#ifdef __MINGW32__
   #define _inline __inline
   #include <include/Wspiapi.h>
   #undef _inline
#else
   #include <Wspiapi.h>
#endif

#include <windows.h>

#include "socket_sys.h"

namespace Falcon {
namespace Sys {

//================================================
// Generic system dependant
//================================================

bool init_system()
{
   WSADATA data;
   if ( WSAStartup( MAKEWORD( 2, 2 ), &data ) != 0 )
      return false;
   return true;
}



bool isIPV4( const String &ipv4__ )
{
   String ipv4 = ipv4__;
   struct addrinfo hints;
   struct addrinfo *res = 0;

   // we want to see an IPv4:
   memset( &hints, 0, sizeof( hints ) );

   hints.ai_family = PF_INET;
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

   hints.ai_family = PF_INET6;
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
   char *errordesc;
   switch( error )
   {
      case -1: errordesc = "(internal) No valid target addresses for selected protocol";
      case WSABASEERR: errordesc = "No Error"; break;
      case WSAEINTR: errordesc = "Interrupted system call"; break;
      case WSAEBADF: errordesc = "Bad file number"; break;
      case WSAEACCES: errordesc = "Permission denied"; break;
      case WSAEFAULT: errordesc = "Bad address"; break;
      case WSAEINVAL: errordesc = "Invalid argument"; break;
      case WSAEMFILE: errordesc = "Too many open files"; break;
      case WSAEWOULDBLOCK: errordesc = "Operation would block"; break;
      case WSAEINPROGRESS: errordesc = "Operation now in progress"; break;
      case WSAEALREADY: errordesc = "Operation already in progress"; break;
      case WSAENOTSOCK: errordesc = "Socket operation on non-socket"; break;
      case WSAEDESTADDRREQ: errordesc = "Destination address required"; break;
      case WSAEMSGSIZE: errordesc = "Message too long"; break;
      case WSAEPROTOTYPE: errordesc = "Protocol wrong type for socket"; break;
      case WSAENOPROTOOPT: errordesc = "Bad protocol option"; break;
      case WSAEPROTONOSUPPORT: errordesc = "Protocol not supported"; break;
      case WSAESOCKTNOSUPPORT: errordesc = "Socket type not supported"; break;
      case WSAEOPNOTSUPP: errordesc = "Operation not supported on socket"; break;
      case WSAEPFNOSUPPORT: errordesc = "Protocol family not supported"; break;
      case WSAEAFNOSUPPORT: errordesc = "Address family not supported by protocol family"; break;
      case WSAEADDRINUSE: errordesc = "Address already in use"; break;
      case WSAEADDRNOTAVAIL: errordesc = "Can't assign requested address"; break;
      case WSAENETDOWN: errordesc = "Network is down"; break;
      case WSAENETUNREACH: errordesc = "Network is unreachable"; break;
      case WSAENETRESET: errordesc = "Net dropped connection or reset"; break;
      case WSAECONNABORTED: errordesc = "Software caused connection abort"; break;
      case WSAECONNRESET: errordesc = "Connection reset by peer"; break;
      case WSAENOBUFS: errordesc = "No buffer space available"; break;
      case WSAEISCONN: errordesc = "Socket is already connected"; break;
      case WSAENOTCONN: errordesc = "Socket is not connected"; break;
      case WSAESHUTDOWN: errordesc = "Can't send after socket shutdown"; break;
      case WSAETOOMANYREFS: errordesc = "Too many references, can't splice"; break;
      case WSAETIMEDOUT: errordesc = "Connection timed out"; break;
      case WSAECONNREFUSED: errordesc = "Connection refused"; break;
      case WSAELOOP: errordesc = "Too many levels of symbolic links"; break;
      case WSAENAMETOOLONG: errordesc = "File name too long"; break;
      case WSAEHOSTDOWN: errordesc = "Host is down"; break;
      case WSAEHOSTUNREACH: errordesc = "No Route to Host"; break;
      case WSAENOTEMPTY: errordesc = "Directory not empty"; break;
      case WSAEPROCLIM: errordesc = "Too many processes"; break;
      case WSAEUSERS: errordesc = "Too many users"; break;
      case WSAEDQUOT: errordesc = "Disc Quota Exceeded"; break;
      case WSAESTALE: errordesc = "Stale NFS file handle"; break;
      case WSASYSNOTREADY: errordesc = "Network SubSystem is unavailable"; break;
      case WSAVERNOTSUPPORTED: errordesc = "WINSOCK DLL Version out of range"; break;
      case WSANOTINITIALISED: errordesc = "Successful WSASTARTUP not yet performed"; break;
      case WSAEREMOTE: errordesc = "Too many levels of remote in path"; break;
      case WSAHOST_NOT_FOUND: errordesc = "Host not found"; break;
      case WSATRY_AGAIN: errordesc = "Non-Authoritative Host not found"; break;
      case WSANO_RECOVERY: errordesc = "Non-Recoverable errors: FORMERR, REFUSED, NOTIMP"; break;
      case WSANO_DATA: errordesc = "Valid name, no data record of requested type"; break;
      default: errordesc = "Unknown error";
      }

   ret = errordesc;
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
   hints.ai_family = PF_UNSPEC;

   //toCString is guaranteed to stay as long as the string exists.
	char hostBuf[256];
   char serviceBuf[64];
   m_host.toCString( hostBuf, 255 );
   m_service.toCString( serviceBuf, 63 );
   int error = getaddrinfo( hostBuf, serviceBuf, &hints, &res ) ;
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

static int s_select( SOCKET skt, int32 msec, int32 mode )
{
   TIMEVAL tv, *tvp;
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

static int s_select_connect( SOCKET skt, int32 msec )
{
   TIMEVAL tv, *tvp;
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

Socket::~Socket()
{
   // ungraceful close.
   terminate();
}

void Socket::terminate()
{
   if ( d.m_iSystemData == 0 )
      return;

   SOCKET s = (SOCKET) d.m_iSystemData;
   closesocket( s );
   d.m_iSystemData = 0;
}

int Socket::readAvailable( int32 msec, const Sys::SystemData *sysData )
{
   m_lastError = 0;
   int count = s_select( (SOCKET) d.m_iSystemData, msec, 0 );
   if ( count == SOCKET_ERROR ) {
      m_lastError = WSAGetLastError();
      return false;
   }
   return count > 0;
}

int Socket::writeAvailable( int32 msec, const Sys::SystemData *sysData )
{
   m_lastError = 0;
   int count = s_select( (SOCKET) d.m_iSystemData, msec, 1 );
   if ( count == SOCKET_ERROR ) {
      m_lastError = WSAGetLastError();
      return false;
   }
   return count > 0;
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
   SOCKET skt = INVALID_SOCKET;
   // find a suitable address.
   int entryId;
   int type = packet ? SOCK_DGRAM : SOCK_STREAM;

   for ( entryId = 0; entryId < addr.getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)addr.getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == PF_INET ) {
         skt = socket( ai->ai_family, type, ai->ai_protocol );
         if ( skt != INVALID_SOCKET ) {
            break;
         }
      }
   }

   if ( skt == INVALID_SOCKET ) {
      m_lastError = -1;
      return false;
   }

   // dispose of old socket
   if ( d.m_iSystemData != 0 ) {
      closesocket( (SOCKET) d.m_iSystemData );
      d.m_iSystemData = 0;
   }

   // need of a broadcast semantic?
   if ( broadcast )
   {
      int iOpt = 1;
      setsockopt( skt, SOL_SOCKET, SO_BROADCAST, (const char *) &iOpt, sizeof( iOpt ));
   }

   // Always useful
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

   m_lastError = WSAGetLastError();
   return false;
}

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

   d.m_iSystemData = 0;
}

TCPSocket::~TCPSocket()
{
}

int32 TCPSocket::recv( byte *buffer, int32 size )
{
   if ( ! readAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int read = ::recv( (SOCKET) d.m_iSystemData, (char *)buffer, size, 0 );
   if ( read < 0 ) {
      m_lastError = WSAGetLastError();
      return -1;
   }

   return read;
}

int32 TCPSocket::send( const byte *buffer, int32 size )
{
   if ( ! writeAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int sent = ::send( (SOCKET) d.m_iSystemData, (char *) buffer, size, 0 );
   if ( sent < 0 ) {
      m_lastError = WSAGetLastError();
      return -1;
   }

   return sent;
}

bool TCPSocket::closeRead()
{
   if ( shutdown( (SOCKET ) d.m_iSystemData, SD_RECEIVE ) == SOCKET_ERROR )
   {
      m_lastError = WSAGetLastError();
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
   if ( shutdown( (SOCKET ) d.m_iSystemData, SD_SEND ) == SOCKET_ERROR ) {
      m_lastError = WSAGetLastError();
      return false;
   }
   // no wait
   return true;
}


bool TCPSocket::close()
{
   if ( shutdown( (SOCKET ) d.m_iSystemData, SD_BOTH ) == SOCKET_ERROR )
   {
      m_lastError = WSAGetLastError();
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
   ULONG mode = 0;

   m_lastError = 0;

   // let's try to connect the addresses in where.
   if ( where.getResolvedCount() == 0 ) {
      if ( ! where.resolve() ) {
         m_lastError = where.m_lastError;
         return false;
      }
   }

   struct addrinfo *ai = 0;
   SOCKET skt = INVALID_SOCKET;
   // find a suitable address.
   int entryId;
   for ( entryId = 0; entryId < where.getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)where.getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == AF_INET ) {
         skt = socket( ai->ai_family, ai->ai_socktype, ai->ai_protocol );
         if ( skt != INVALID_SOCKET ) {
            break;
         }
      }
   }

   // quite impossible but...
   if ( skt == INVALID_SOCKET ) {
      m_lastError = -1;
      return false;
   }

   // dispose of old socket
   if ( d.m_iSystemData != 0 ) {
      closesocket( (SOCKET) d.m_iSystemData );
      d.m_iSystemData = 0;
   }

   BOOL bOptVal = TRUE;

   // set keepalive
   if ( setsockopt( skt, SOL_SOCKET, SO_KEEPALIVE, (char *) &bOptVal, sizeof( BOOL ) ) == SOCKET_ERROR )
   {
      m_lastError = WSAGetLastError();
      return false;
   }

   // if timeout is -1, do not set nonblocking.
   if( m_timeout >= 0 ) {
      // set nonblocking
      mode = 1;
      ioctlsocket( skt,  FIONBIO, &mode );
   }

   d.m_iSystemData = skt;
   m_lastError = 0;
   m_connected = false;
   int res = ::connect( skt, ai->ai_addr, ai->ai_addrlen );

   // reset nonblocking status
   if ( m_timeout >= 0 ) {
      mode = 0;
      ioctlsocket( skt,  FIONBIO, &mode );
   }

   where.getResolvedEntry( entryId, m_address.m_host, m_address.m_service, m_address.m_port );

   if ( res == SOCKET_ERROR ) {
      m_lastError = WSAGetLastError();

      if( m_lastError != WSAEWOULDBLOCK )
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
         m_lastError = WSAGetLastError();
      }
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

   int status = s_select_connect( (SOCKET) d.m_iSystemData, m_timeout );
   if ( status == 1 ) {
      m_connected = true;
      return true;
   }
   else if( status == -1 )
      m_lastError = WSAGetLastError();
   else
      m_lastError = 0;

   return false;
}


//================================================
// Server
//================================================

ServerSocket::ServerSocket( bool ipv6 ):
   m_bListening( false ),
   Socket( 0, ipv6 )
{
}

ServerSocket::~ServerSocket()
{
}

TCPSocket *ServerSocket::accept()
{
   SOCKET srv = (SOCKET) d.m_iSystemData;

   if ( ! m_bListening ) {
      if ( ::listen( srv, SOMAXCONN ) != 0 ) {
         m_lastError = WSAGetLastError();
         return 0;
      }
      m_bListening = true;
   }

   if ( s_select( srv, m_timeout, 0 ) ) {
      int addrlen;
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

      SOCKET skt = ::accept( srv, address, &addrlen );
      TCPSocket *s = new TCPSocket( (void *)skt );

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
   int family = ipv6? PF_INET6 : PF_INET;
   SOCKET skt = socket( ipv6, SOCK_DGRAM, 0 );
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
   SOCKET skt = (SOCKET) d.m_iSystemData;
   setsockopt( skt, SOL_SOCKET, SO_BROADCAST, (const char *) &iOpt, sizeof( iOpt ));
}

int32 UDPSocket::recvFrom( byte *buffer, int32 size, Address &data )
{
   SOCKET s = (SOCKET) d.m_iSystemData;
   // sockaddr_in6 should be the longest possible structure we may receive.
   struct sockaddr_in6 addr;
   struct sockaddr *paddr = (struct sockaddr *) &addr;
   socklen_t len = sizeof( addr );

   if ( ! readAvailable( m_timeout ) ) {
      if ( m_lastError != 0 )
         return -1; // error
      return -2; // timed out
   }

   int32 retsize = ::recvfrom( s, (char *)buffer, size, 0, paddr, &len );

   if ( retsize == -1 )
      m_lastError = WSAGetLastError();
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
   SOCKET s = (SOCKET) d.m_iSystemData;

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

   int32 retsize = ::sendto( s, (char *)buffer, size, 0, ai->ai_addr, ai->ai_addrlen );

   if ( retsize == -1 )
      m_lastError = WSAGetLastError();
   else {
      m_lastError = 0;
   }

   return retsize;
}

} // namespace
}

/* end of socket_sys_win.cpp */
