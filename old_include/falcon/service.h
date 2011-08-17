/*
   FALCON - The Falcon Programming Language.
   FILE: service.h

   Service declaration
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun feb 13 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Service and service provider classes.
*/

#ifndef flc_service_H
#define flc_service_H

#include <falcon/string.h>

namespace Falcon {

/** Service class.
   Falcon modules are meant to interact with OS, databases, web servers, and many other
   resources in behalf of Falcon scripts.

   However, some of the resources handled by Falcon binary modules may be extremely
   useful also for embedding application. Instead of re-linking or re-abstracting the required
   low level services, this interface allows the modules to publish a set of functionalities
   that can be used by both the Falcon scripts and the embedding application. This also
   grants a somehow more direct interface between applications and scripts, that can share
   objects representing low level resources as files, database connections, shared memory,
   internet sockets and so on.

   Service are required to a module by the Falcon::Module::getService(). a module
   will publish its services via Falcon::Module::publishService(). Applications willing
   to create many service instance in a row may find more efficient to use
   Falcon::Module::getServiceProvider(), which returns a factory for the given service.

   As Falcon modules are objects that should usually be loaded via DLL interface, a service will
   declare all of its methods as virtual (except for const inline methdos), so that access to
   methods by the user application won't require any linkage with the falcon modules. Of course,
   this causes a minor inefficience in function calls.

   The module where services are defined should declare 
   \code
      #define FALCON_EXPORT_SERVICE
   \endcode
   before any Falcon file is inlcuded (specifically, before falcon/setup.h). This will tell
   MS-Windows compilers that the module is willing to offer the service to the users of the target
   DLL. Subclasses derived from services should maintain the FALCON_SERVICE signature to provide
   the same DLL export policy (necessary in MSVC C++ projects).


   \note Willing to spare every bit of CPU on module side for non-required overhead, the FALCON_FUNC
      that interacts with the script may call directly the needed function as C calls, passing the
      pointer to the service to them just as a data holder. The Service instance will hold the required
      data plus the set of pointers to the functions that are to be published. Methods may then be
      just inline calls to those function pointers; this will still require no linkage for modules into
      applications, and will have the same speed as virtual method calls, but you'll spare extra
      virtual calls inside the module.
*/

class FALCON_SERVICE Service {
   String m_name;

public:
   /** Creates the service assigning it a certain name.
      The service requries a name by which it can be published by the module.
   */
   Service( const String & name );

   /** Destructor needs to be virtual. */
   virtual ~Service();

   const String &getServiceName() const { return m_name; }
};

}

#endif

/* end of service.h */
