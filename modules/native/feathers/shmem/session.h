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
#include <falcon/item.h>

namespace Falcon
{

class Symbol;
class VMContext;
class Storer;
class Restorer;

/** Automatism to implement persistent data across multiple runs of a program.
 *
 */
class Session
{
public:
   /** Creates an empty session.
    *
    * When created in this form, the session won't be given any ID
    * nor expiration time.
    *
    * ID and expiration time must be provided separately before the
    * session can be stored.
    */
   Session();

   /** Creates a session, and possibly set an expiration time.
    *
    * This constructor creates a session and possibly immediately assigns
    * an expiration time to it.
    *
    * Notice that it's preferably to let the session start to expire as soon
    * as the session is first saved on a storage.
    *
    */
   Session( const String& id, int64 to = 0 );

   /** Destroys the session */
   ~Session();

   /** Starts the session as a new session right now. */
   void start();

   /** Gets the ID of the session.
    * \param target the string where the ID of the session is stored.
    *
    * \note The method is threadsafe (this is why you have to provide
    * a target string where to place the ID).
    */
   void getID( String& target ) const;

   /** Sets an ID for the session.
    *
    * \note The method is threadsafe.
    */
   void setID( const String& n );

   /** Adds a symbol (with its value) to the session recording. */
   void addSymbol(Symbol* sym, const Item& value=Item());

   /** Removes a symbol from the session recording */
   bool removeSymbol(Symbol* sym);

   /** Records the required symbols from the given context. */
   void record(VMContext* ctx);

   /** Updates the given context with the values stored in the session. */
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
    * Restores the session from an already loaded restorer.
    *
    * The given restorer must have been already loaded via
    * Restorer::restore.
    */
   void restore(Restorer* restorer);

   /** Gets the value stored in the session for a symbol. */
   bool get(Symbol* sym, Item& value) const;
   /** Gets the value stored in the session for a symbol. */
   bool get(const String& symName, Item& value) const;

   /** Returns the timestamp when the start() method was called. */
   int64 createdAt() const;
   /** Returns the timestamp when the session expires */
   int64 expiresAt() const;

   /** Returns the timeout set for the session.
    *
    * The timeout is the amount of time that is added to the expiration
    * time each time the session is refreshed.
    */
   int64 timeout() const;

   /** Changes the timeout (and refreshes the session).
    *
    * If the timeout is set to zero, the session never expires.
    */
   void timeout( int64 to );

   /** Extends the session lifetime of the given timeout.
    *
    * If the session has a timeout, the expiration time is set
    * to the current moment in time plus the given timeout; otherwise
    * this call has no effect.
    *
    */
   void tick();

   /** Checks if a session is currently expired. */
   bool isExpired() const;

   void gcMark( uint32 mark );
   uint32 currentMark() const { return m_mark; }

private:
   /** Not cloneable */
   Session(const Session& ) {}

   String m_id;
   int64 m_tsCreation;
   int64 m_tsExpire;
   int64 m_timeout;
   mutable bool m_bExpired;
   uint32 m_mark;

   class Private;
   Private* _p;
};

}

#endif

/* end of session.h */
