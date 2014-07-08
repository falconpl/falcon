/*
   FALCON - The Falcon Programming Language.
   FILE: socket.cpp

   The socket module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "inet_mod.h"
#include "inet_fm.h"
#include <falcon/module.h>

#if WITH_OPENSSL
#include <openssl/ssl.h> // check for OPENSSL_NO_SSL2
#endif // WITH_OPENSSL

/*#
   @module inet Low level IP networking.
   @ingroup feathers
   @brief Low level TCP/IP networking support.

   The socket module provides a low level access to network (TCP/IP) socket wise
   functions. UDP and TCP protocol are supported, and the module provides also name
   resolution facilities, both performed automatically when calling connect and
   bind methods, or manually by calling an appropriate name or address resolution
   routine.

   The module supports both IPv4 and IPv6 networking; generally, IPv6 is chosen
   transparently when an IPv6 address is provided or retrieved by the name
   resolution system, if the host system supports it.

   The Socket module defines a @a NetError class that is raised on network errors. The
   class is derived from core IoError and doesn't add any method or property.

   @note The module can be loaded using the command
      @code
      load socket
      @endcode

   @beginmodule feathers.inet
*/

FALCON_MODULE_DECL
{
   if ( ! Falcon::Mod::init_system() )
   {
      return 0;
   }

   Falcon::Module *self = new Falcon::Feathers::ModuleInet();
   return self;
}

/* end of socket.cpp */
/* vim: set ai et sw=3 ts= sts=3: */
