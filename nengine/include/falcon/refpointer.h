/*
   FALCON - The Falcon Programming Language.
   FILE: refpointer.h

   Automatically manages reference-count oriented datatype.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 09 Jan 2011 13:37:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_REFPOINTER_H_
#define FALCON_REFPOINTER_H_

#include <falcon/mt.h>

namespace Falcon {

/** Automatically manages reference-count oriented datatype.
*/

template<class __T>
class FALCON_DYN_CLASS ref_ptr
{

public:
   ref_ptr():
   m_data(0),
   m_rc(0)
   {}

   ref_ptr( __T* data ):
      m_data(data),
      m_rc( new Refcount )
   {
   }

   ref_ptr(const ref_ptr<__T>& rp):
      m_data( rp.m_data ),
      m_rc( rp.m_rc )
   {
      if( m_rc ) m_rc->incref();
   }

   ~ref_ptr() {
      if( m_rc != 0 ) {
         if ( m_rc->decref() == 0 )
         {
            delete m_rc;
            delete m_data;
         }
      }
   }

   __T& operator*() { return *m_data; }
   __T* operator->() { return m_data; }

  ref_ptr<__T>& operator = (const ref_ptr<__T>& rp)
  {
      if (this != &rp) // Avoid self assignment
      {
         if ( m_rc != 0 )
         {
            if ( m_rc->decref() == 0 )
            {
               delete m_rc;
               delete m_data;
            }
         }

         m_data = rp.m_data;
         m_rc = rp.m_rc;
         if (m_rc) m_rc->incref();
     }
  }
  
  ref_ptr<__T>& operator = ( __T* data )
  {
      if ( m_rc != 0 )
      {
         if ( m_rc->decref() == 0 )
         {
            delete m_rc;
            delete m_data;
         }
      }

      m_data = data;
      m_rc = new Refcount;
  }


private:

   class Refcount
   {
      int32 m_count;

   public:
      Refcount():
         m_count(1)
      {}

      ~Refcount() {}

      void incref() { atomicInc(m_count); }
      int32 decref() { return atomicDec(m_count); }
   };

   __T* m_data;
   Refcount* m_rc;
};

}

#endif /* FALCON_REFPOINTER_H_ */
