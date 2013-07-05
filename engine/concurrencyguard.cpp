/*
   FALCON - The Falcon Programming Language.
   FILE: concurrencyguard.cpp

   Guard against unguarded concurrent access to a shared object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Apr 2013 16:29:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/concurrencyguard.cpp"

#include <falcon/concurrencyguard.h>
#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>
#include <falcon/process.h>
#include <falcon/fassert.h>

namespace Falcon {

ConcurrencyGuard::ConcurrencyGuard():
      m_readCount(0),
      m_writeCount(0)
{}


ConcurrencyGuard::Token ConcurrencyGuard::write( VMContext* ctx )
{
   // we first set the write count;
   if( atomicInc(m_writeCount) > 1 || atomicFetch(m_readCount) > 0 )
   {
      // set another writer count to prevent any other try to succeed.
      atomicInc(m_writeCount);
      Error* error = new ConcurrencyError(ErrorParam(e_concurrence,__LINE__, SRC));
      ctx->process()->terminateWithError(error);
      ctx->terminate();
      throw error;
   }

   return toeknWrite;
}

ConcurrencyGuard::Token ConcurrencyGuard::read( VMContext* ctx )
{
   atomicInc(m_readCount);
   if( atomicFetch(m_writeCount) > 0 )
   {
      // set another writer count to prevent any other try to succeed.
      atomicInc(m_writeCount);
      Error* error = new ConcurrencyError(ErrorParam(e_concurrence,__LINE__, SRC));
      ctx->process()->terminateWithError(error);
      ctx->terminate();
      throw error;
   }


   return toeknRead;
}

void ConcurrencyGuard::releaseWrite()
{
   atomicDec(m_writeCount);
   fassert(m_writeCount >= 0);
}

void ConcurrencyGuard::releaseRead()
{
   atomicDec(m_readCount);
   fassert(m_readCount >= 0);
}

}

/* end of concurrencyguard.h */
