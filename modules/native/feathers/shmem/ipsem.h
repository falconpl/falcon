/*
   FALCON - The Falcon Programming Language.
   FILE: ipsem.h

   Inter-process semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Nov 2013 16:27:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FEATHERS_IPSEM_H_
#define _FALCON_FEATHERS_IPSEM_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {

/** Multiplatform abstraction of an inter-process semaphore.
 *
 */
class IPSem
{
public:
   /** Creates a semaphore that needs initialization. */
   IPSem();

   /** Creates a semaphore and tries a non-exclusive creation. */
   IPSem( const String& name );

   /** Copy another semaphore by trying to open the same semaphore as that. */
   IPSem( const IPSem& other );

   ~IPSem();

   typedef enum {
      e_om_open,
      e_om_openex,
      e_om_create
   }
   t_open_mode;

   void init( const String& name, t_open_mode mode );
   void open(const String& name) { init(name, e_om_open ); }
   void openExisting(const String& name) { init(name, e_om_openex ); }
   void create(const String& name) { init(name, e_om_create ); }

   void close( bool bDelete = false );

   void post();
   bool wait( int64 to = -1 );

   bool tryWait() { return wait(0); }

private:
   class Private;
   Private* _p;
};

}

#endif /* _FALCON_FEATHERS_IPSEM_H_ */
