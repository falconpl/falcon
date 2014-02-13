/*
   FALCON - The Falcon Programming Language.
   FILE: session_srv.h

   Service exposing the session through DLL interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FEATHERS_SHMEM_SESSION_SRV_H_
#define _FALCON_FEATHERS_SHMEM_SESSION_SRV_H_

#include <falcon/gclock.h>
#include <falcon/storer.h>
#include <falcon/service.h>

#define SESSIONSERVICE_NAME  "Session"

namespace Falcon
{

class Session;

/** Service exposing sessions.
 *
 */
class SessionService: public Service
{
public:


   SessionService( Module* creator );
   virtual ~SessionService();

   virtual void itemize( Item& target ) const;

   virtual const String& id() const;
   virtual void begin() const;
   virtual void getID( String& target ) const;
   virtual void setID( const String& n ) const;
   virtual void setOpenMode_file() const;
   virtual void setOpenMode_shmem() const;
   virtual void setOpenMode_shmem_bu() const;

   virtual void addSymbol(const Symbol* sym, const Item& value=Item()) const;
   virtual bool removeSymbol(const Symbol* sym) const;
   virtual void record(VMContext* ctx) const;
   virtual void apply(VMContext* ctx) const;
   virtual void save( VMContext* ctx ) const;
   virtual void load( VMContext* ctx, bool bApply = false ) const;
   virtual void close() const;
   virtual bool get(const Symbol* sym, Item& value) const;
   virtual bool get(const String& symName, Item& value) const;
   virtual int64 createdAt() const;
   virtual int64 expiresAt() const;
   virtual void open() const;
   virtual void create() const;
   virtual int64 timeout() const;
   virtual void timeout( int64 to ) const;
   virtual void tick() const;
   virtual bool isExpired() const;
   virtual void commitStore(VMContext* ctx, Storer* sto ) const;
   virtual bool checkLoad() const;

private:
   Session* m_session;
   GCLock* m_sessionLock;
};

}

#endif

/* end of session_srv.h */
