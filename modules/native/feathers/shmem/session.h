/*
   FALCON - The Falcon Programming Language.
   FILE: session.h

   Falcon script interface for Inter-process semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FEATHERS_SHMEM_SESSION_H_
#define _FALCON_FEATHERS_SHMEM_SESSION_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

class Symbol;
class VMContext;

class Session
{
public:
   Session();
   Session( const String& name );
   ~Session();

   const String& name() const { return m_name; }
   void name( const String& n ) { m_name = n; }

   void addSymbol(Symbol* sym, const Item& value=Item());
   bool removeSymbol(Symbol* sym);

   void record(VMContext* ctx);
   void apply(VMContext* ctx) const;

   /**
    * Prepares a storer to commit the session data.
    *
    * After this call susccessfully returns, the storer can
    * be stored on a file via storer::commit.
    *
    * The method never goes deep in ctx, but it might prepare
    * ctx to perform some deep operation after the method returns.
    */
   void store(VMContext* ctx, Storer* storer) const;

   /**
    * Restores the session from an alaredy loaded restorer.
    *
    * The given restorer must have been already loaded via
    * Restorer::restore.
    */
   void restore(Restorer* restorer);

private:
   String m_name;

   class Private;
   Private* _p;
};

}

#endif

/* end of session.h */
