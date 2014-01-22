/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_refcount.h

   SQLite3 driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_DBI_REFCOUNT_H
#define FALCON_DBI_REFCOUNT_H

namespace Falcon {

template<class _T>
class DBIRefCounter
{
public:
   DBIRefCounter( const _T& handler ):
       m_Handler(handler),
       m_nRefCount(1)
   {}
   

   void incref() { m_nRefCount ++; }
   void decref() { if ( --m_nRefCount == 0 ) delete this; }
   const _T& handle() const { return m_Handler; }
   _T& handle() { return m_Handler; }

protected:
   virtual ~DBIRefCounter() {
   }

private:
   _T m_Handler;
   int m_nRefCount;
};

}

#endif

/* end of dbi_refcount.h */

