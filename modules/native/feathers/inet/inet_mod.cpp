/*
   FALCON - The Falcon Programming Language.
   FILE: inet_mod.cpp

   BSD socket generic basic support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

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

#include <netdb.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include "inet_mod.h"
#include "inet_ext.h"


#include <falcon/stderrors.h>
#include <falcon/autocstring.h>
#include <falcon/engine.h>
#include <falcon/stdmpxfactories.h>


namespace Falcon {
namespace Mod {

//================================================
// Generic system dependant
//================================================

bool init_system()
{
   return true;
}

bool shutdown_system()
{
   return true;
}

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


//================================================
// Address asynchronous resolver
//================================================

Resolver::Resolver( ContextManager* mgr, const Class* handler ):
         Shared( mgr, handler ),
         m_address(0),
         m_thread(0)
{
}

void Resolver::startResolving( Address* address )
{
   fassert( m_address == 0 );
   address->incref();
   m_address = address;
   m_thread = new SysThread(new TH(this) );
   bool result = m_thread->start(ThreadParams().detached(true).stackSize(32*1024));

   // we expect the thread to start
   if( ! result ) {
      throw FALCON_SIGN_XERROR( CodeError, e_start_thread, .sysError(m_thread->lastError()) );
   }
}

Resolver::~Resolver()
{
   if( m_address != 0 )
   {
      m_address->decref();
   }
}


Resolver::TH::TH( Resolver* res ):
         m_res(res)
{
   res->incref();
}

Resolver::TH::~TH()
{
   m_res->decref();
}

void* Resolver::TH::run()
{
   m_res->address()->resolve();
   m_res->signal();

   delete this;
   return 0;
}

//================================================
// Address
//================================================

Address::~Address()
{
   clear();
}

Address::Address( const String& addr )
{
   if( ! parse(addr) )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError, FALSOCK_ERR_ADDRESS, .desc(FALSOCK_ERR_ADDRESS_MSG));
   }
}

Address::Address( const String& host, const String& service ):
      m_systemData(0),
      m_resolvCount(0),
      m_activeHostId(-1),
      m_mark(0)
{
   m_host = host;
   m_service = service;
   int64 port = 0;
   if( service.parseInt(port) )
   {
      m_port = (int) port;
   }
}

Address::Address( const String& host, int32 port ):
      m_systemData(0),
      m_resolvCount(0),
      m_activeHostId(-1),
      m_mark(0)
{
   m_host = host;
   m_port = port;
   m_service.N(port);
}

Address::Address( const Address& other ):
      m_systemData(0),
      m_resolvCount(0),
      m_activeHostId(-1),
      m_mark(0)
{
   m_host = other.m_host;
   m_port = other.m_port;
   m_service = other.m_service;
}


void Address::clear()
{
   if( m_systemData != 0 )
   {
      struct addrinfo *res = (struct addrinfo *) m_systemData;
      ::freeaddrinfo(res);
      m_systemData = 0;

   }
   m_resolvCount = 0;
   m_activeHostId = -1;
}

size_t Address::occupiedMemory() const
{
   return sizeof( Mod::Address ) + 16
            + (getResolvedCount() * sizeof(struct sockaddr_in) );
}

void Address::set( const String &host, const String &service )
{
   clear();
   m_host.bufferize( host );
   m_service.bufferize( service );
   int64 port;
   if( m_service.parseInt(port) )
   {
      m_port = port;
   }
   else {
      port = 0;
   }
}


bool Address::parse( const String& tgt )
{
   int64 portValue = 0;

   if( m_systemData != 0 )
   {
      struct addrinfo *res = (struct addrinfo *) m_systemData;
      ::freeaddrinfo(res);
      m_systemData = 0;
   }

   m_port = 0;
   if( tgt.empty() )
   {
      m_host.clear();
      m_service.clear();
      return true;
   }

   // remove extra spaces
   const String *addr;
   String temp;
   if( String::isWhiteSpace(tgt.getCharAt(0)) || String::isWhiteSpace(tgt.getCharAt(tgt.length()-1)) )
   {
      temp = tgt;
      addr = &temp;
      temp.trim();
   }
   else {
      addr = &tgt;
   }

   // dot/quad or host:port format?
   length_t pos = addr->find('[');
   if( pos == String::npos )
   {
      // one ":" accepted only
      pos = addr->find(':');
      length_t pos1 = addr->rfind(':');
      if( pos != pos1 || pos == addr->length() )
      {
         return false;
      }

      // just an host?
      if( pos == String::npos )
      {
         m_host = *addr;
         m_service.clear();
      }
      else {
         m_host = addr->subString(0,pos);
         m_service = addr->subString(pos+1);

         // actually not needed, but it's nice to have.
         if( m_service.parseInt(portValue) )
         {
            m_port = (int) portValue;
         }
      }
      return true;
   }

   // in ipv6, [...] must be at beginning.
   if( pos != 0 )
   {
      return false;
   }

   // ipv6 [...]:...
   length_t posEnd = addr->find(']');
   length_t posService = addr->rfind(':');
   if( posEnd == String::npos )
   {
      return false;
   }

   // do we have just the host? "[...]"
   if( posEnd +1 == addr->length() )
   {
      // ipv6 host only
      m_host = addr->subString(1,addr->length()-2);
      m_service.clear();
      return true;
   }

   // the last : must be right after the ] "[...]:service"
   if( posService != posEnd + 1 )
   {
      return false;
   }

   m_host = addr->subString(1,posEnd-1);
   m_service = addr->subString(posService + 1);
   // actually not needed, but it's nice to have.
   if( m_service.parseInt(portValue) )
   {
      m_port = (int) portValue;
   }
   return true;
}


bool Address::resolve( bool useNameResolver )
{
   char hostBuf[256];
   char serviceBuf[64];
   m_host.toCString( hostBuf, 255 );
   m_service.toCString( serviceBuf, 63 );

   struct addrinfo hints;
   struct addrinfo *res = 0;

   memset( &hints, 0, sizeof( hints ) );
   hints.ai_family = AF_UNSPEC;

   if( ! useNameResolver )
   {
      hints.ai_flags = AI_NUMERICHOST;
   }

   int error = ::getaddrinfo( hostBuf, serviceBuf, &hints, &res ) ;
   if ( error != 0 ) {
      m_lastError = (int64) error;
      return false;
   }

   if ( m_systemData != 0 )
   {
      struct addrinfo *oldres = (struct addrinfo *) m_systemData;
      ::freeaddrinfo( oldres );
   }

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


void Address::errorDesc( String& target )
{
   const char* ed = gai_strerror((int) m_lastError);
   target.fromUTF8(ed);
}

bool Address::isResolved() const
{
   return this->getResolvedCount() != 0;
}


void Address::toString( String& str ) const
{
   if( m_host.find(':') != String::npos )
   {
      str = "[" + m_host + "]";
   }
   else {
      str = m_host;
   }

   if( ! m_service.empty() )
   {
      str +=":" +m_service;
   }
   else if ( m_port != 0 )
   {
      str += ":";
      str.N(m_port);
   }
}

//================================================
// Socket
//================================================

Socket::~Socket()
{
   this->close();

   #if WITH_OPENSSL
   if ( m_sslData != 0 )
   {
      delete m_sslData;
      m_sslData = 0;
   }
   #endif
}


void Socket::bind( Address *addr, bool dgram )
{
   if ( m_type != e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   // has the address to be resolved?
   if( ! addr->isResolved() ) {
      if( ! addr->resolve(false) ) {
         throw FALCON_SIGN_XERROR(Ext::NetError,
                        FALSOCK_ERR_UNRESOLVED, .desc(FALSOCK_ERR_UNRESOLVED_MSG));
      }
   }

   // try to bind to the resovled host
   struct addrinfo *ai = 0;
   int skt = -1;
   // find a suitable address.
   int entryId;
   int type = m_type == dgram ? SOCK_DGRAM: SOCK_STREAM;

   for ( entryId = 0; entryId < addr->getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)addr->getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == AF_INET ) {
         skt = socket( ai->ai_family, type, ai->ai_protocol );
         if ( skt > 0 ) {
            break;
         }
      }
   }

   if ( skt == -1 ) {
      throw FALCON_SIGN_XERROR( Ext::NetError,
                              FALSOCK_ERR_CREATE, .desc(FALSOCK_ERR_CREATE_MSG)
                              .sysError((uint32) errno ));
   }

   // always useful
   {
      int iOpt = 1;
      int res = ::setsockopt( skt, SOL_SOCKET, SO_REUSEADDR, (const char *) &iOpt, sizeof( iOpt ));
      if (res != 0)
      {
         throw FALCON_SIGN_XERROR( Ext::NetError,
                                 FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
                                 .extra("setsockopt")
                                 .sysError((uint32) errno ));

      }
   }

   int res = ::bind( skt, ai->ai_addr, ai->ai_addrlen );

   if ( res == 0 )
   {
      // success!!!
      m_type = dgram ? e_type_dgram : e_type_server;
      m_skt = skt;
      m_boundFamily = ai->ai_family;
      if( m_address != 0 )
      {
         // shouldn't happen, however...
         m_address->decref();
      }

      m_address = addr;
      addr->incref();
   }

   throw FALCON_SIGN_XERROR( Ext::NetError,
               FALSOCK_ERR_BIND, .desc(FALSOCK_ERR_BIND_MSG)
               .sysError((uint32) errno ));
}



void Socket::close()
{
   if( m_skt >= 0 )
   {
      if ( ::close( (int) m_skt ) == 0 )
      {
         ::close(m_skt); // for safety
         m_skt = -1;
         m_type = e_type_none;
      }
      else {
         throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_CLOSE, .desc(FALSOCK_ERR_CLOSE_MSG)
                       .sysError((uint32) errno ));
      }
   }
}

void Socket::closeWrite()
{
   if( m_skt >= 0 )
   {
      if ( ::shutdown( (int) m_skt, SHUT_WR ) == 0 )
      {
         ::close(m_skt); // for safety
         m_skt = -1;
         m_type = e_type_none;
      }
      else {
         throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_CLOSE, .desc(FALSOCK_ERR_CLOSE_MSG)
                       .sysError((uint32) errno ));
      }
   }
}

void Socket::closeRead()
{
   if( m_skt >= 0 )
   {
      if ( ::shutdown( (int) m_skt, SHUT_RD ) == 0 )
      {
         ::close(m_skt); // for safety
         m_skt = -1;
         m_type = e_type_none;
      }
      else {
         throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_CLOSE, .desc(FALSOCK_ERR_CLOSE_MSG)
                       .sysError((uint32) errno ));
      }
   }
}

void Socket::connect( Address* where, bool async )
{
   if ( m_type == e_type_server )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE,
               .desc(FALSOCK_ERR_INCOMPATIBLE_MSG)
               );
   }

   int flags = 0;

   // let's try to connect the addresses in where.
   if ( ! where->isResolved() )
   {
      if( ! where->resolve(false) )
      {
            throw FALCON_SIGN_XERROR(Ext::NetError,
                     FALSOCK_ERR_UNRESOLVED, .desc(FALSOCK_ERR_UNRESOLVED_MSG));
      }
   }

   struct addrinfo *ai = 0;
   int skt = -1;
   // find a suitable address.
   int entryId;
   for ( entryId = 0; entryId < where->getResolvedCount(); entryId++ )
   {
      ai = (struct addrinfo *)where->getHostSystemData( entryId );
      if ( m_ipv6 || ai->ai_family == AF_INET ) {
         skt = socket( ai->ai_family, SOCK_STREAM, ai->ai_protocol );
         if ( skt < 0 )
         {
            throw FALCON_SIGN_XERROR( Ext::NetError,
                          FALSOCK_ERR_CREATE, .desc(FALSOCK_ERR_CREATE_MSG)
                          .sysError((uint32) errno ));
         }
        break;
      }
   }

   // set keepalive
   /*
   int bOptVal = 1;
   if ( setsockopt( skt, SOL_SOCKET, SO_KEEPALIVE, (char *) &bOptVal, sizeof( int ) ) < 0 )
   {
      m_lastError = errno;
      return false;
   }
   */

   int32 oldflags = 0;
   m_skt = skt;
   if( async )
   {
      oldflags = getFcntl(F_GETFL);
      oldflags |= O_NONBLOCK;
      setFcntl(F_SETFL, flags);
   }

   m_bConnected = false;
   int res = ::connect( skt, ai->ai_addr, ai->ai_addrlen );

   if ( res < 0 )
   {
      int error = errno;
      if( error != EINPROGRESS )
      {
         ::close( skt );
         m_skt = -1;
         throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_CREATE, .desc(FALSOCK_ERR_CREATE_MSG)
                       .sysError( error ));
      }
   }
   else {
      m_bConnected = true;
   }

   m_skt = skt;
   // if the connection is VERY fast, we may get connected even if we were async.
   if( async )
   {
      // reset the non-blocking status
      setFcntl(F_SETFL, oldflags);
   }

   if( m_type != e_type_dgram )
   {
      m_type = e_type_stream;
   }
}

bool Socket::isConnected() const
{
   if( m_bConnected )
   {
      return m_bConnected;
   }
   else if(m_type == e_type_stream )
   {
      if( ::connect(m_skt, 0, 0) == EAGAIN )
      {
         return false;
      }

      m_bConnected = true;
   }
   return m_bConnected;
}

int Socket::getFcntl( int option ) const
{
   int result = ::fcntl( m_skt, option );
   if( result == -1 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
                 FALSOCK_ERR_FCNTL, .desc(FALSOCK_ERR_FCNTL_MSG)
                 .sysError((uint32) errno ));
   }

   return option;
}

int Socket::setFcntl( int option, int flags ) const
{
   int res = ::fcntl( m_skt, option, flags );
   if( res == -1 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_FCNTL, .desc(FALSOCK_ERR_FCNTL_MSG)
                       .sysError((uint32) errno ));
   }

   return res;
}

void Socket::fcntlFlagsOn( int flags ) const
{
   int cflags = this->getFcntl( F_GETFL );
   cflags |= flags;
   this->setFcntl(F_SETFL, cflags);
}

void Socket::fcntlFlagsOff( int flags ) const
{
   int cflags = this->getFcntl( F_GETFL );
   cflags &= ~flags;
   this->setFcntl(F_SETFL, cflags);
}

void Socket::setNonBlocking( bool mode ) const
{
   int flags = this->getFcntl( F_GETFL );

   if( mode )
   {
      flags |= O_NONBLOCK;
   }
   else {
      flags &= ~O_NONBLOCK;
   }

   this->setFcntl(F_SETFL, flags);
}


/** Throws net error on error. */
void Socket::getBoolOption( int level, int option, bool& value )
{
   if ( m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   int res = 0;
   unsigned int len = 0;

   if( ::getsockopt( m_skt, level, option, &res, &len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("getsockopt")
            .sysError((uint32) errno ));
   }

   value = (res != 0);
}


void Socket::getIntOption( int level, int option, int& value )
{
   if ( m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   value = 0;
   unsigned int len = 0;

   if( ::getsockopt( m_skt, level, option, &value, &len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("getsockopt")
            .sysError((uint32) errno ));
   }
}


void Socket::getStringOption( int level, int option, String& value )
{
   if ( m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   char buffer[512];
   unsigned int len = 512;

   if( ::getsockopt( m_skt, level, option, buffer, &len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("getsockopt")
            .sysError((uint32) errno ));
   }

   value.fromUTF8( buffer, len );
}

/** Throws net error on error. */
void Socket::setBoolOption( int level, int option, bool value )
{
   if ( m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   int res = value ? 1 : 0;
   socklen_t len = sizeof(int);

   if( ::setsockopt( m_skt, level, option, &res, len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("setsockopt")
            .sysError((uint32) errno ));
   }
}

/** Throws net error on error. */
void Socket::setIntOption( int level, int option, int value )
{
   if ( m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   socklen_t len = sizeof(int);

   if( ::setsockopt( m_skt, level, option, &value, len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("setsockopt")
            .sysError((uint32) errno ));
   }
}


void Socket::setStringOption( int level, int option, const String& value )
{
   if ( m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   AutoCString cval(value);
   socklen_t len = cval.length();

   if( ::setsockopt( m_skt, level, option, cval.c_str(), len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("setsockopt")
            .sysError((uint32) errno ));
   }
}


//================================================
// Server
//================================================

void Socket::setServerMode( Address* local, int listenBacklog )
{
   if ( m_type != e_type_none )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   if( listenBacklog < 0 )
   {
      listenBacklog = SOMAXCONN;
   }

   bind(local, false);

   if ( ::listen( m_skt, SOMAXCONN ) != 0 )
   {
      uint32 error = errno;
      ::close(m_skt);
      m_skt = -1;
      m_type = e_type_none;

      throw FALCON_SIGN_XERROR(Ext::NetError,
                     FALSOCK_ERR_LISTEN, .desc(FALSOCK_ERR_LISTEN_MSG)
                     .sysError(error));
   }

   m_type = e_type_server;
}


Socket *Socket::accept()
{
   if ( m_type != e_type_server )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
               FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

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

   int skt = ::accept( m_skt, address, &addrlen );
   if( skt < 0 )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
                     FALSOCK_ERR_ACCEPT, .desc(FALSOCK_ERR_ACCEPT_MSG)
                     .sysError((uint32) errno));
   }

   char hostName[64];
   char servName[64];
   Address* ask = new Address;
   if ( getnameinfo( address, addrlen, hostName, 63, servName, 63, NI_NUMERICHOST | NI_NUMERICSERV ) == 0 )
   {
      ask->set(hostName,servName);
   }

   Socket *s = new Socket( skt, ask, m_boundFamily == AF_INET6 );
   ask->decref();

   return s;
}

int32 Socket::recv( byte *buffer, int32 size, Address *data )
{
   if( m_type == e_type_server || m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
                 FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG)
                 .sysError((uint32) errno ));
   }

   int32 retsize = 0;

   if( data != 0 )
   {
      // sockaddr_in6 should be the longest possible structure we may receive.
      struct sockaddr_in6 addr;
      struct sockaddr *paddr = (struct sockaddr *) &addr;
      socklen_t len = sizeof( addr );

      retsize = ::recvfrom( m_skt, buffer, size, 0, paddr, &len );
      if( retsize > 0 )
      {
         char host[64];
         char serv[31];
         int error = ::getnameinfo( paddr, len, host, 63, serv, 31, NI_NUMERICHOST | NI_NUMERICSERV );
         if ( error == 0 ) {
            data->set( host, serv );
         }
         else {
            data->lastError( (uint32) error );
         }
      }
   }
   else
   {
      retsize = ::recv( m_skt, buffer, size, 0 );
   }

   if( retsize < 0 )
   {
      if( errno == EWOULDBLOCK || errno == EAGAIN )
      {
         return -1;
      }

      throw FALCON_SIGN_XERROR( Ext::NetError,
                 FALSOCK_ERR_RECV, .desc(FALSOCK_ERR_RECV_MSG)
                 .sysError((uint32) errno ));
   }

   return retsize;
}


int32 Socket::send( const byte *buffer, int32 size, Address *where )
{
   if( m_type == e_type_server || m_type == e_type_none )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
                 FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG)
                 .sysError((uint32) errno ));
   }

   int32 retsize = 0;

   // let's try to connect the addresses in where.
   if ( where != 0 )
   {
      if( ! where->isResolved() )
      {
         if ( ! where->resolve(false) ) {
            throw FALCON_SIGN_XERROR( Ext::NetError,
                  FALSOCK_ERR_UNRESOLVED, .desc(FALSOCK_ERR_UNRESOLVED_MSG)
               );
         }
      }

      struct addrinfo *ai = 0;

      // find a suitable address.
      int entryId;
      for ( entryId = 0; entryId < where->getResolvedCount(); entryId++ )
      {
         ai = (struct addrinfo *)where->getHostSystemData( entryId );
         if ( m_ipv6 || ai->ai_family == AF_INET ) {
            break;
         }
      }

      // quite impossible but...
      fassert( entryId != where->getResolvedCount() );

      retsize = ::sendto( m_skt, buffer, size, 0, ai->ai_addr, ai->ai_addrlen );
   }
   else {
      retsize = ::send( m_skt, buffer, size, 0 );
   }

   if( retsize < 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
                 FALSOCK_ERR_SEND, .desc(FALSOCK_ERR_SEND_MSG)
                 .sysError((uint32) errno ));
   }

   return retsize;
}

//===============================================
// Socket Selectable Socket
//===============================================


SocketSelectable::SocketSelectable( const Class* cls, Socket* inst ):
      FDSelectable(cls, inst)
{
   inst->incref();
}

SocketSelectable::~SocketSelectable()
{
   Socket* socket = static_cast<Socket*>(instance());
   socket->decref();
}

const Multiplex::Factory* SocketSelectable::factory() const
{
   return Engine::instance()->stdMpxFactories()->fileDataMpxFact();
}

int SocketSelectable::getFd() const
{
   Socket* socket = static_cast<Socket*>(instance());
   return socket->descriptor();
}


//===============================================
// SSL Extensions
//===============================================

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


SSLData::ssl_error_t Socket::sslConfig( bool asServer,
                                           SSLData::sslVersion_t sslVer,
                                           const char* certFile,
                                           const char* pkeyFile,
                                           const char* certAuthFile )
{
   Falcon::Mod::ssl_init();

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
   case SSLData::SSLv2:  sslD->sslMethod = (SSL_METHOD*) SSLv2_method(); break;
#endif
   case SSLData::SSLv3:  sslD->sslMethod = (SSL_METHOD*) SSLv3_method(); break;
   case SSLData::SSLv23: sslD->sslMethod = (SSL_METHOD*) SSLv23_method(); break;
   case SSLData::TLSv1:  sslD->sslMethod = (SSL_METHOD*) TLSv1_method(); break;
   case SSLData::DTLSv1: sslD->sslMethod = (SSL_METHOD*) DTLSv1_method(); break;
   default: sslD->sslMethod = (SSL_METHOD*) SSLv3_method();
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
   if ( ( i = SSL_set_fd( sslD->sslHandle, m_skt ) ) != 1 )
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

SSLData::ssl_error_t Socket::sslConnect()
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
   {
      return SSLData::notconnected_error;
   }

   if ( m_sslData->asServer ) // server-side socket
   {
      i = SSL_accept( m_sslData->sslHandle );
   }
   else // client-side socket
   {
      i = SSL_connect( m_sslData->sslHandle );
   }

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

int32 Socket::sslWrite( const byte* buf, int32 sz )
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

int32 Socket::sslRead( byte* buf, int32 sz )
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

//================================================
// SSLData
//================================================
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

} // namespace
}

/* end of inet_mod.cpp */
