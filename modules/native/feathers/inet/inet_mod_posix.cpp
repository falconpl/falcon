/*
   FALCON - The Falcon Programming Language.
   FILE: inet_mod_posix.cpp

   BSD socket generic basic support -- POSIX specific
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#define SRC "modules/native/feathres/inet/inet_mod_posix.cpp"

#include "inet_mod.h"
#include "inet_fm.h"

#include <falcon/stdmpxfactories.h>

namespace Falcon {
namespace Mod {

bool init_system()
{
   return true;
}

bool shutdown_system()
{
   return true;
}


static int s_getFcntl( FALCON_SOCKET skt, int option )
{
   int result = ::fcntl( skt, option );
   if( result == -1 )
   {
      throw FALCON_SIGN_XERROR( Feathers::NetError,
                 FALSOCK_ERR_FCNTL, .desc(FALSOCK_ERR_FCNTL_MSG)
                 .sysError((uint32) errno ));
   }

   return option;
}

static int s_setFcntl( FALCON_SOCKET skt, int option, int flags )
{
   int res = ::fcntl( skt, option, flags );
   if( res == -1 )
   {
      throw FALCON_SIGN_XERROR( Feathers::NetError,
                       FALSOCK_ERR_FCNTL, .desc(FALSOCK_ERR_FCNTL_MSG)
                       .sysError((uint32) errno ));
   }

   return res;
}


void Socket::setNonBlocking( bool mode ) const
{
   int flags = s_getFcntl( m_skt, F_GETFL );

   if( mode )
   {
      flags |= O_NONBLOCK;
   }
   else {
      flags &= ~O_NONBLOCK;
   }

   s_setFcntl( m_skt, F_SETFL, flags);
}

bool Socket::isNonBlocking() const
{
   int flags = s_getFcntl( m_skt, F_GETFL );
   return (flags & O_NONBLOCK) != 0;
}

int Socket::sys_getsockopt( int level, int option_name, void *option_value, FALCON_SOCKLEN_AS_INT * option_len) const
{
   return ::getsockopt( m_skt, level, option_name, option_value, option_len );
}

int Socket::sys_setsockopt( int level, int option_name, const void *option_value, FALCON_SOCKLEN_AS_INT option_len) const
{
   return ::setsockopt( m_skt, level, option_name, option_value, option_len );
}



const Multiplex::Factory* SocketSelectable::factory() const
{
   // this on POSIX:
   return Engine::instance()->stdMpxFactories()->fileDataMpxFact();
}


}
}

/* end of inet_mod_posix.cpp */
