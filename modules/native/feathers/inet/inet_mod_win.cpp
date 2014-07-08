/*
   FALCON - The Falcon Programming Language.
   FILE: inet_mod_win.cpp

   BSD socket generic basic support -- Windows specific
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#define SRC "modules/native/feathres/inet/inet_mod_win.cpp"



#include "inet_mod.h"
#include "inet_fm.h"

#ifdef __MINGW32__
   #define _inline __inline
   #include<ws2spi.h>//#include <include/Wspiapi.h>
   #undef _inline
#else
   #include <Wspiapi.h>
#endif


namespace Falcon {
namespace Mod {

bool getErrorDesc( int64 error, String &ret )
{
   const char *errordesc;
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
// Generic system dependant
//================================================

bool init_system()
{
   WSADATA data;
   memset( &data, 0, sizeof(data));
   if ( WSAStartup( MAKEWORD( 2, 2 ), &data ) != 0 )
      return false;
   return true;
}

bool shutdown_system()
{
   WSACleanup();
   return true;
}


void Socket::setNonBlocking( bool mode ) const
{
   u_long iMode = mode ? 1 : 0;

   int iResult = ioctlsocket( m_skt, FIONBIO, &iMode);
   if (iResult != NO_ERROR)
   {
       throw FALCON_SIGN_XERROR( Feathers::NetError,
                 FALSOCK_ERR_FCNTL, .desc(FALSOCK_ERR_FCNTL_MSG)
                 .sysError((uint32) errno ));
   }

   m_bioMode = mode;
}

bool Socket::isNonBlocking() const
{
   return m_bioMode;
}

int Socket::sys_getsockopt( int level, int option_name, void *option_value, FALCON_SOCKLEN_T * option_len) const
{
   return ::getsockopt( m_skt, level, option_name, (char*) option_value, option_len );
}

int Socket::sys_setsockopt( int level, int option_name, const void *option_value, FALCON_SOCKLEN_T option_len) const
{
   return ::setsockopt( m_skt, level, option_name, (const char*) option_value, option_len );
}



const Multiplex::Factory* SocketSelectable::factory() const
{
   Feathers::ModuleInet* mi = static_cast<Feathers::ModuleInet*>(handler()->module());
   return mi->selectMPXFactory();
}


}
}

/* end of inet_mod_win.cpp */
