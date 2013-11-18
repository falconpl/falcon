/*
   FALCON - The Falcon Programming Language.
   FILE: session.cpp

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

class Session
{
public:
   Session();
   Session( const String& name );
   ~Session();

   const String& name() const { return m_name; }
   void name( const String& n ) { m_name = n; }

   void addSymbol(Symbol* sym);
   bool removeSymbol(Symbol* sym);

   void record( Module* mod);
   void apply( Module* mod) const;

private:
   String m_name;

   class Private;
   Private* _p;
};

}

#endif

/* end of session.cpp */
