/*
   FALCON - The Falcon Programming Language.
   FILE: flc_refcount.h

   Reference count system.
   This is not intendended for VM or API usage, but only for
   internal FALCON compiler(s) usage.
   VM has special handling of refcounting when needed.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer giu 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
      Reference count system.
   This is not intendended for VM or API usage, but only for
   internal FALCON compiler(s) usage.
   VM has special handling of refcounting when needed.
*/

#ifndef FALCON_REFCOUNTER_H
#define FALCON_REFCOUNTER_H

namespace Falcon
{


template<class T>
class Refcounter
{
   T m_item;
   int m_count;

public:
   Refcounter():
      m_count(0)
   {}

   virtual ~Refcounter() {}

   /** Creator.
      Sets the count to zero. The creator must incref, it it wishes.
   */
   Refcounter( const T &val ):
      m_item( val ),
      m_count(0)
   {}

   Refcounter( const Refcounter &source ):
      m_item( source.m_item ),
      m_count( 0 )
   {}


   void incref() { m_count++; }
   void decref() { m_count--; if ( m_count <= 0 ) delete  this ; }

   T &access() { return m_item; }

   T &operator *() { return m_item; }
};

}

#endif
/* end of flc_refcount.h */
