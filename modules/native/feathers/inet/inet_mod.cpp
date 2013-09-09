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

Address::Address( const String& addr)
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

void Address::convertToIPv6( void *vai, void* vsock6, socklen_t& sock6len )
{
   struct addrinfo *ai = (struct addrinfo *) vai;

   char host[256];
   strcpy(host, "::ffff:");
   char serv[32];
   int res = ::getnameinfo( ai->ai_addr, ai->ai_addrlen, host+7, 255-7, serv, 31, NI_NUMERICHOST | NI_NUMERICSERV );

   if ( res != 0 )
   {
      const char* ed = gai_strerror(res);
      throw FALCON_SIGN_XERROR(Ext::NetError, FALSOCK_ERR_RESOLV, .desc(FALSOCK_ERR_RESOLV_MSG)
               .extra(String("").N(res).A(": ").A(ed)) );
   }

   struct addrinfo *resolved = 0;
   struct addrinfo hints;
   memset(&hints, 0, sizeof(hints) );
   hints.ai_family = AF_INET6;
   hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV | AI_PASSIVE;

   res = ::getaddrinfo( host, serv, &hints, &resolved );
   if ( res != 0 )
   {
      const char* ed = gai_strerror(res);
      throw FALCON_SIGN_XERROR(Ext::NetError, FALSOCK_ERR_RESOLV, .desc(FALSOCK_ERR_RESOLV_MSG)
               .extra(String("").N(res).A(": ").A(ed)) );
   }

   sock6len = resolved->ai_addrlen;
   memcpy( vsock6, resolved->ai_addr, resolved->ai_addrlen );
   ::freeaddrinfo( resolved );
}



bool Address::getResolvedEntry( int32 count, String &entry, String &service, int32 &port, int32& family )
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
   {
      error = getnameinfo( res->ai_addr, res->ai_addrlen, host, 255, serv, 31, NI_NUMERICHOST | NI_NUMERICSERV );
   }

   if ( error == 0 )
   {
      entry.bufferize( host );
      service.bufferize( serv );
      // port is in the same position for ip and ipv6
      struct sockaddr_in *saddr = (struct sockaddr_in *)res->ai_addr;
      port = ntohs( saddr->sin_port );
      family = saddr->sin_family;
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


void* Address::getRandomHostSystemData( int32 family ) const
{
   int32 count = 0;

   struct addrinfo *res = (struct addrinfo *) m_systemData;
   while( res != 0 )
   {
     if( family == AF_UNSPEC || res->ai_family == family )
     {
        ++count;
     }
     res = res->ai_next;
   }

   if( count == 0 )
   {
      return 0;
   }

   if( count > 1 )
   {
      count = (Engine::instance()->mtrand().randInt() % count)+1;
   }

   res = (struct addrinfo *) m_systemData;
   while( res != 0 )
   {
     if( family == AF_UNSPEC || res->ai_family == family )
     {
        if( --count == 0 )
        {
           break;
        }
     }
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

Socket::Socket():
     m_family(-1),
     m_type(-1),
     m_protocol(-1),

     m_address(0),
     m_bConnected( false ),
     m_mark(0)
     #if WITH_OPENSSL
     ,m_sslData( 0 )
     #endif
{
   m_skt = -1;
}


Socket::Socket(int family, int type, int protocol):
     m_family(-1),
     m_type(-1),
     m_protocol(-1),

     m_address(0),
     m_bConnected( false ),
     m_mark(0)
     #if WITH_OPENSSL
     ,m_sslData( 0 )
     #endif
{
   m_skt = -1;
   create(family, type, protocol);
}

Socket::Socket( FALCON_SOCKET socket, int family, int type, int protocol, Address* remote ):
     m_family(family),
     m_type(type),
     m_protocol(protocol),
     m_address( remote ),
     m_bConnected( true ),
     m_skt(socket),
     m_mark(0)
     #if WITH_OPENSSL
     ,m_sslData( 0 )
     #endif
{
  if( remote != 0 )
  {
     remote->incref();
  }
}

void Socket::create( int family, int type, int protocol )
{
   if( m_skt != -1 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_ALREADY_CREATED, .desc(FALSOCK_ERR_ALREADY_CREATED_MSG));
   }


   m_family = family;
   m_type = type;
   m_protocol = protocol;
   m_skt = ::socket(family, type, protocol);
   if( m_skt == -1 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_CREATE, .desc(FALSOCK_ERR_CREATE_MSG));
   }

}

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


void* Socket::getValidAddress( Address* addr ) const
{
   // has the address to be resolved?
   if( ! addr->isResolved() ) {
      if( ! addr->resolve(false) ) {
         throw FALCON_SIGN_XERROR(Ext::NetError,
                        FALSOCK_ERR_UNRESOLVED, .desc(FALSOCK_ERR_UNRESOLVED_MSG));
      }
   }

   struct addrinfo *ai = (struct addrinfo *)addr->getRandomHostSystemData( m_family );
   // upgrade ipv4 addresses into ipv6.
   if( ai == 0 && m_family == AF_INET6 )
   {
      ai = (struct addrinfo *)addr->getRandomHostSystemData( AF_INET );
      if ( ai != 0 )
      {
         struct addrinfo ai6;
         struct sockaddr_storage storage;

         Address::convertToIPv6( ai, &storage, ai6.ai_addrlen );
         ai6.ai_addr = (struct sockaddr*) &storage;
         ai = &ai6;
      }
   }

   if( ai == 0 )
   {
      throw FALCON_SIGN_XERROR(Ext::NetError,
                              FALSOCK_ERR_INCOMPATIBLE, .desc(FALSOCK_ERR_INCOMPATIBLE_MSG));
   }

   return ai;
}

void Socket::bind( Address *addr )
{
   // try to bind to the resolved host
   struct addrinfo *ai = (struct addrinfo *) getValidAddress( addr );

   // find a suitable address.
   int res = ::bind( m_skt, ai->ai_addr, ai->ai_addrlen );

   if ( res == 0 )
   {
      // success!!!
      if( m_address != 0 )
      {
         // shouldn't happen, however...
         m_address->decref();
      }

      m_address = addr;
      addr->incref();
   }
   else {
      throw FALCON_SIGN_XERROR( Ext::NetError,
               FALSOCK_ERR_BIND, .desc(FALSOCK_ERR_BIND_MSG)
               .sysError((uint32) errno ));
   }
}


void Socket::close()
{
   if( m_skt >= 0 )
   {
      if ( ::close( (int) m_skt ) == 0 )
      {
         ::close(m_skt); // for safety
         m_skt = -1;
         m_type = -1;
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
      }
      else {
         throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_CLOSE, .desc(FALSOCK_ERR_CLOSE_MSG)
                       .sysError((uint32) errno ));
      }
   }
}


bool Socket::broadcasting() const
{
   return this->getBoolOption( PF_INET, SO_BROADCAST );
}


void Socket::broadcasting( bool mode )
{
   this->setBoolOption( PF_INET, SO_BROADCAST, mode );
}


void Socket::closeRead()
{
   if( m_skt >= 0 )
   {
      if ( ::shutdown( (int) m_skt, SHUT_RD ) != 0 )
      {
         ::close(m_skt); // for safety
         m_skt = -1;
         m_type = -1;
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
   int flags = 0;

   // try to bind to the resolved host
   struct addrinfo *ai = (struct addrinfo *) getValidAddress( where );
   fassert(ai != 0);

   int32 oldflags = 0;
   if( async )
   {
      oldflags = getFcntl(F_GETFL);
      oldflags |= O_NONBLOCK;
      setFcntl(F_SETFL, flags);
   }

   m_bConnected = false;
   int res = ::connect( m_skt, ai->ai_addr, ai->ai_addrlen );

   if ( res < 0 )
   {
      int error = errno;
      if( error != EINPROGRESS )
      {
         ::close( m_skt );
         m_skt = -1;
         throw FALCON_SIGN_XERROR( Ext::NetError,
                       FALSOCK_ERR_CREATE, .desc(FALSOCK_ERR_CREATE_MSG)
                       .sysError( error ));
      }
   }
   else {
      m_bConnected = true;
   }

   // if the connection is VERY fast, we may get connected even if we were async.
   if( async )
   {
      // reset the non-blocking status
      setFcntl(F_SETFL, oldflags);
   }
}

bool Socket::isConnected() const
{
   if( m_bConnected )
   {
      return m_bConnected;
   }
   else
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

bool Socket::isNonBlocking() const
{
   int flags = this->getFcntl( F_GETFL );
   return (flags & O_NONBLOCK) != 0;
}


/** Throws net error on error. */
bool Socket::getBoolOption( int level, int option) const
{
   int res = 0;
   unsigned int len = sizeof(int);

   if( ::getsockopt( m_skt, level, option, &res, &len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("getsockopt")
            .sysError((uint32) errno ));
   }

   return res != 0;
}


int  Socket::getIntOption( int level, int option ) const
{
   int value = 0;
   unsigned int len = sizeof(int);

   if( ::getsockopt( m_skt, level, option, &value, &len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("getsockopt")
            .sysError((uint32) errno ));
   }

   return value;
}


void Socket::getStringOption( int level, int option, String& value ) const
{
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


void Socket::getDataOption( int level, int option, void* data, size_t& data_len ) const
{
   socklen_t len = (socklen_t) data_len;

   if( ::getsockopt( m_skt, level, option, data, &len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
           FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
           .extra("getsockopt")
           .sysError((uint32) errno ));
   }

   data_len = (size_t) len;
}

/** Throws net error on error. */
void Socket::setBoolOption( int level, int option, bool value )
{
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


void Socket::setDataOption( int level, int option, const void* data, size_t data_len )
{
   if( ::setsockopt( m_skt, level, option, data, (socklen_t) data_len ) != 0 )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError,
            FALSOCK_ERR_GENERIC, .desc(FALSOCK_ERR_GENERIC_MSG)
            .extra("setsockopt")
            .sysError((uint32) errno ));
   }
}



void Socket::listen( int listenBacklog )
{
   if( listenBacklog < 0 )
   {
      listenBacklog = SOMAXCONN;
   }

   if ( ::listen( m_skt, SOMAXCONN ) != 0 )
   {
      uint32 error = errno;
      ::close(m_skt);
      m_skt = -1;
      m_type = -1;

      throw FALCON_SIGN_XERROR(Ext::NetError,
                     FALSOCK_ERR_LISTEN, .desc(FALSOCK_ERR_LISTEN_MSG)
                     .sysError(error));
   }
}


Socket *Socket::accept()
{
   socklen_t addrlen;
   struct sockaddr *address;
   struct sockaddr_storage addrIn6;

   address = (struct sockaddr *) &addrIn6;
   addrlen = sizeof( addrIn6 );


   int skt = ::accept( m_skt, address, &addrlen );
   if( skt < 0 )
   {
      // account for spurious wakeups
      if( errno == EAGAIN || errno == EINPROGRESS )
      {
         return 0;
      }

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

   Socket *s = new Socket( skt, address->sa_family, SOCK_STREAM, m_protocol, ask );
   ask->decref();

   return s;
}

int32 Socket::recv( byte *buffer, int32 size, Address *data )
{
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
      // account for spurious wake ups in multiplexers
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
   int32 retsize = 0;

   // let's try to connect the addresses in where.
   if ( where != 0 )
   {
      struct addrinfo *ai = (struct addrinfo *) getValidAddress( where );
      fassert(ai != 0);
      retsize = ::sendto( m_skt, buffer, size, 0, ai->ai_addr, ai->ai_addrlen );
   }
   else {
      retsize = ::send( m_skt, buffer, size, 0 );
   }

   if( retsize < 0 )
   {
      // account for spurious wakeups
      if( errno == EAGAIN || errno == EINPROGRESS )
      {
         return -1;
      }

      throw FALCON_SIGN_XERROR( Ext::NetError,
                 FALSOCK_ERR_SEND, .desc(FALSOCK_ERR_SEND_MSG)
                 .sysError((uint32) errno ));
   }

   return retsize;
}

Socket::t_option_type Socket::getOptionType( int level, int option )
{
   t_option_type param_type = param_unknown;

   switch( level )
   {
   case SOL_SOCKET:
      switch(option)
      {
      case SO_DEBUG: param_type = param_bool; break;
      case SO_REUSEADDR: param_type = param_bool; break;
      case SO_TYPE: param_type = param_int; break;
      case SO_ERROR : param_type = param_int; break;
      case SO_DONTROUTE: param_type = param_bool; break;
      case SO_BROADCAST: param_type = param_bool; break;
      case SO_SNDBUF: param_type = param_int; break;
      case SO_RCVBUF: param_type = param_int; break;
      case SO_KEEPALIVE: param_type = param_bool; break;
      case SO_OOBINLINE: param_type = param_bool; break;
      case SO_LINGER: param_type = param_linger; break;
      case SO_RCVLOWAT: param_type = param_int; break;
      case SO_SNDLOWAT: param_type = param_int; break;
      case SO_RCVTIMEO: param_type = param_int; break;
      case SO_SNDTIMEO: param_type = param_int; break;
      }
      break;

   case IPPROTO_IP:
      switch(option)
      {

      }
      break;

   case IPPROTO_IPV6:
       switch(option)
       {
       case IPV6_JOIN_GROUP: param_type = param_data; break;
       case IPV6_LEAVE_GROUP: param_type = param_data; break;
       case IPV6_MULTICAST_HOPS: param_type = param_int; break;
       case IPV6_MULTICAST_IF: param_type = param_string; break;
       case IPV6_MULTICAST_LOOP: param_type = param_bool; break;
       case IPV6_UNICAST_HOPS: param_type = param_int; break;
       case IPV6_V6ONLY: param_type = param_bool; break;
       }
       break;

   case IPPROTO_RAW:
        switch(option)
        {

        }
        break;

   case IPPROTO_TCP:
       switch(option)
       {

       }
       break;

   case IPPROTO_UDP:
      switch(option)
      {

      }
      break;
   }

   return param_type;
}

void Socket::setOption( int level, int option, const Item& value)
{
   t_option_type param_type = Socket::getOptionType(level, option);

   switch( param_type )
   {
   case param_bool: setBoolOption( level, option, value.isTrue() ); break;

   case param_int:
      if( ! value.isOrdinal() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type,
                  .extra("Level/option combination requires an integer value"));
      }
      setIntOption( level, option, value.forceInteger() );
      break;

   case param_string:
      if( ! value.isString() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type,
                  .extra("Level/option combination requires a string value"));
      }
      setStringOption( level, option, *value.asString() );
      break;

   case param_data:
      if( ! value.isString() && ! value.isNil() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type,
                  .extra("Level/option combination requires a string or nil value"));
      }

      if( value.isNil() )
      {
         setDataOption( level, option, 0, 0 );
      }
      else {
         setDataOption( level, option, value.asString()->getRawStorage(), value.asString()->size() );
      }
      break;

   case param_linger:
      if( ! value.isInteger() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type,
                  .extra("Linger option requires an integer value"));
      }
      else {
         struct linger ling;
         int64 vl = (int) value.asInteger();
         if( vl < 0 )
         {
            ling.l_onoff = 0;
            ling.l_linger = 0;
         }
         else {
            ling.l_onoff = 1;
            ling.l_linger = (int) vl;
         }
         setDataOption( level, option, &ling, sizeof(ling) );
      }
      break;

   case param_unknown:
      if( ! value.isInteger() && ! value.isBoolean() && ! value.isNil() )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type,
                  .extra("Unknown level/option combination requires an integer, boolean or nil value"));
      }
      else
      {
         int fv = 0;
         if( value.isInteger() ) {
            fv = value.isInteger();
         }
         else if( value.isTrue() )
         {
            fv = 1;
         }
         setIntOption( level, option, fv );
      }
      break;
   }
}

void Socket::getOption( int level, int option, Item& value ) const
{
   t_option_type param_type = Socket::getOptionType(level, option);

   switch( param_type )
   {
   case param_bool: value.setBoolean(getBoolOption( level, option )); break;

   case param_int:
   case param_unknown:
      value.setInteger( getIntOption( level, option ) );
      break;

   case param_string:
      {
         String* string = new String;
         getStringOption( level, option, *string );
         value = FALCON_GC_HANDLE( string );
      }
      break;

   case param_data:
      {
         String* string = new String;
         string->reserve(512);
         length_t len = 512;
         getDataOption( level, option, string->getRawStorage(), len );
         string->size(len);
         value = FALCON_GC_HANDLE( string );
      }
      break;

   case param_linger:
      {
         struct linger ling;
         ling.l_onoff = 0;
         ling.l_linger = 0;
         length_t len = sizeof(ling);
         getDataOption( level, option, &ling, len );
         if( ling.l_onoff == 0 )
         {
            value.setInteger( -1 );
         }
         else {
            value.setInteger( (int64) ling.l_linger );
         }
      }
      break;
   }
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


void Socket::sslConfig( bool asServer,
                   SSLData::sslVersion_t sslVer,
                   String* certFile,
                   String* pkeyFile,
                   String* certAuthFile )
{
   Falcon::Mod::ssl_init();

   if ( m_skt < 0 ) // no socket to attach
   {
      throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
           .desc( FALSOCK_ERR_SSLCONFIG_MSG )
           .extra( "Not ready" ) );
   }

   // called before?
   if ( m_sslData ) {
      return;
   }

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

   try
   {
      // create context
      sslD->sslContext = SSL_CTX_new( sslD->sslMethod );
      if ( !sslD->sslContext )
      {
         throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
                         .desc( FALSOCK_ERR_SSLCONFIG_MSG )
                         .extra( "Cannot create the SSL context" ) );
      }

      // certificate file
      if ( certFile && ! certFile->empty() )
      {
         AutoCString scf(*certFile);
         if ( ( i = SSL_CTX_use_certificate_file( sslD->sslContext, scf.c_str(),
             SSL_FILETYPE_PEM ) != 1 ) )
         {
            throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
                 .desc( FALSOCK_ERR_SSLCONFIG_MSG )
                 .extra( "Failed to use certificate file" ) );

         }
         sslD->certFile.bufferize(*certFile);
      }

      // private key file
      if ( pkeyFile && ! pkeyFile->empty() )
      {
         AutoCString scf(*pkeyFile);
         if ( ( i = SSL_CTX_use_PrivateKey_file( sslD->sslContext, scf.c_str(),
             SSL_FILETYPE_PEM ) != 1 ) )
         {
            throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
                          .desc( FALSOCK_ERR_SSLCONFIG_MSG )
                          .extra( "Failed to use key file" ) );
         }
         sslD->keyFile.bufferize(*pkeyFile);
      }

      // certificates authorities
      if ( certAuthFile && ! certAuthFile->empty() )
      {
         AutoCString scf(*certAuthFile);
         STACK_OF( X509_NAME ) *cert_names;
         cert_names = SSL_load_client_CA_file( scf.c_str() );
         if ( cert_names != 0 )
         {
            SSL_CTX_set_client_CA_list( sslD->sslContext, cert_names );
         }
         else
         {
            throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
                                   .desc( FALSOCK_ERR_SSLCONFIG_MSG )
                                   .extra( "Failed to use certificate authority file" ) );
         }
         sslD->caFile.bufferize(*certAuthFile);
      }

      // ssl handle
      sslD->sslHandle = SSL_new( sslD->sslContext );
      if ( !sslD->sslHandle )
      {
         throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
                          .desc( FALSOCK_ERR_SSLCONFIG_MSG )
                          .extra( "Cannot create SSL Handle" ) );
      }

      // attach file descriptor
      if ( ( i = SSL_set_fd( sslD->sslHandle, m_skt ) ) != 1 )
      {
         throw new Ext::NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__, SRC )
                          .desc( FALSOCK_ERR_SSLCONFIG_MSG )
                          .extra( "Failed to use attach the file descriptor" ) );
      }

      // done
      m_sslData = sslD;
   }
   catch(...)
   {
      delete sslD;
   }
}

void Socket::sslClear()
{
   if ( m_sslData )
   {
      delete m_sslData;
      m_sslData = 0;
   }
}

void Socket::sslConnect()
{
   //int flags = 0;
   int i;

   // need ssl context
   if ( !m_sslData )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_SSLCONNECT,
                    .desc(FALSOCK_ERR_SSLCONNECT_MSG)
                    .extra( "SSL not configured on this socket" ));
   }


   // no need to call several times
   if ( m_sslData->handshakeState != SSLData::handshake_todo )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_SSLCONNECT,
                    .desc(FALSOCK_ERR_SSLCONNECT_MSG)
                    .extra( "SSL handshake already performed" ));
   }

   // socket needs to be connected
   if ( ! isConnected() )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_SSLCONNECT,
                          .desc(FALSOCK_ERR_SSLCONNECT_MSG)
                          .extra( "SSL not connected" ));
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
      int32 error = SSL_get_error( m_sslData->sslHandle, i );
      if( error == SSL_ERROR_SYSCALL && errno != 0 )
      {
         throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_RECV,
                              .desc(FALSOCK_ERR_RECV_MSG)
                              .sysError((uint32) errno) );
      }
      else if (  error == SSL_ERROR_SSL )
      {
         char buffer[512];
         ERR_error_string_n(error, buffer, 512);
         throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_RECV,
                     .desc(FALSOCK_ERR_RECV_MSG)
                     .extra( String("SSL ").A(buffer)) );
      }
   }

   m_sslData->handshakeState = SSLData::handshake_ok;
}

int32 Socket::sslWrite( const byte* buf, int32 sz )
{
   int i = SSL_write( m_sslData->sslHandle, buf, sz );
   if ( i <= 0 )
   {
      int error = SSL_get_error( m_sslData->sslHandle, i );
      if( error == SSL_ERROR_SYSCALL && errno != 0 )
      {
         throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_RECV,
                              .desc(FALSOCK_ERR_RECV_MSG)
                              .sysError((uint32) errno) );
      }
      else if (  error == SSL_ERROR_SSL )
      {
         char buffer[512];
         ERR_error_string_n(error, buffer, 512);
         throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_RECV,
                     .desc(FALSOCK_ERR_RECV_MSG)
                     .extra( String("SSL ").A(buffer)) );
      }
   }
   return i;
}

int32 Socket::sslRead( byte* buf, int32 sz )
{
   int i = SSL_read( m_sslData->sslHandle, buf, sz );
   if ( i <= 0 )
   {
      int error = SSL_get_error( m_sslData->sslHandle, i );
      if( error == SSL_ERROR_SYSCALL && errno != 0 )
      {
         throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_RECV,
                              .desc(FALSOCK_ERR_RECV_MSG)
                              .sysError((uint32) errno) );
      }
      else if (  error == SSL_ERROR_SSL )
      {
         char buffer[512];
         ERR_error_string_n(error, buffer, 512);
         throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_RECV,
                     .desc(FALSOCK_ERR_RECV_MSG)
                     .extra( String("SSL ").A(buffer)) );
      }

      return 0;
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
