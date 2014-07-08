/*
   FALCON - The Falcon Programming Language.
   FILE: session_srv.cpp

   Service exposing the session through DLL interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/shmem/session_srv.cpp"

#include "session_srv.h"
#include "shmem_fm.h"
#include "session.h"

#include <falcon/engine.h>

namespace Falcon {

SessionService::SessionService( Module* creator ):
   Service( SESSIONSERVICE_NAME, creator )
{
   m_session = new Session();
   Class* cls = static_cast<Feathers::ModuleShmem*>(module())->sessionClass();
   m_sessionLock = Engine::GC_lock(Item(cls, m_session));
}

SessionService::~SessionService()
{
   m_sessionLock->dispose();
}

void SessionService::itemize( Item& target ) const
{
   target.copyFromLocal(m_sessionLock->item());
}


const String& SessionService::id() const
{
   return m_session->id();
}


void SessionService::begin() const
{
   m_session->begin();
}


void SessionService::getID( String& target ) const
{
   m_session->getID(target);
}


void SessionService::setID( const String& n ) const
{
   m_session->setID(n);
}


void SessionService::setOpenMode_file() const
{
   m_session->setOpenMode(Session::e_om_file);
}


void SessionService::setOpenMode_shmem() const
{
   m_session->setOpenMode(Session::e_om_shmem);
}


void SessionService::setOpenMode_shmem_bu() const
{
   m_session->setOpenMode(Session::e_om_shmem_bu);
}


void SessionService::addSymbol(const Symbol* sym, const Item& value) const
{
   m_session->addSymbol(sym, value);
}


bool SessionService::removeSymbol(const Symbol* sym) const
{
   return m_session->removeSymbol(sym);
}


void SessionService::record(VMContext* ctx) const
{
   m_session->record(ctx);
}


void SessionService::apply(VMContext* ctx) const
{
   m_session->apply(ctx);
}


void SessionService::save( VMContext* ctx ) const
{
   m_session->save(ctx);
}


void SessionService::load( VMContext* ctx, bool bApply ) const
{
   m_session->load(ctx,bApply);
}


void SessionService::close() const
{
   m_session->close();
}


bool SessionService::get(const Symbol* sym, Item& value) const
{
   return m_session->get(sym, value);
}


bool SessionService::get(const String& symName, Item& value) const
{
   return m_session->get(symName, value);
}


int64 SessionService::createdAt() const
{
   return m_session->createdAt();
}


int64 SessionService::expiresAt() const
{
   return m_session->expiresAt();
}


void SessionService::open() const
{
   m_session->open();
}


void SessionService::create() const
{
   m_session->create();
}


int64 SessionService::timeout() const
{
   return m_session->timeout();
}

void SessionService::timeout( int64 to ) const
{
   m_session->timeout(to);
}

void SessionService::tick() const
{
   m_session->tick();
}

bool SessionService::isExpired() const
{
   return m_session->isExpired();
}


void SessionService::commitStore(VMContext* ctx, Storer* sto ) const
{
   m_session->commitStore(ctx,sto);
}



bool SessionService::checkLoad() const
{
   return m_session->checkLoad();
}

}

/* end of session_srv.cpp */
