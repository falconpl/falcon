/*
   FALCON - The Falcon Programming Language.
   FILE: corecarrier.h

   Template class to simplify full reflection of application data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 06 Sep 2009 20:29:54 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_CORECARRIER_H
#define FLC_CORECARRIER_H

#include <falcon/setup.h>
#include <falcon/coreobject.h>
#include <falcon/error.h>

namespace Falcon {

template<class _T>
class CoreCarrier: public CoreObject
{
   _T* m_carried;

public:
   CoreCarrier( const CoreClass* base, _T* la ):
      CoreObject( base ),
      m_carried( la )
   {
      if ( la != 0 )
         la->incref();

      setUserData(la); // just not to have it 0
   }

   CoreCarrier( const CoreCarrier& cc ):
      CoreObject( cc ),
      m_carried( cc.m_carried )
   {
      if( m_carried != 0 )
         m_carried->incref();
      setUserData( m_carried ); // just not to have it 0
   }

   virtual ~CoreCarrier()
   {
      if ( m_carried != 0 )
         carried()->decref();
   }

   bool hasProperty( const String &key ) const
   {
      uint32 pos = 0;
      return generator()->properties().findKey(key, pos);
   }

   bool setProperty( const String &prop, const Item &value )
   {
      if ( hasProperty(prop) )
      {
         throw new AccessError( ErrorParam( e_prop_ro, __LINE__ )
                  .origin( e_orig_runtime )
                  .extra( prop ) );
      }

      return false;
   }

   bool getProperty( const String &key, Item &ret ) const
   {
      return defaultProperty( key, ret );
   }


   virtual CoreCarrier *clone() const
   {
      return new CoreCarrier<_T>( *this );
   }

   _T* carried() const { return m_carried; }
   void carried( _T* c )
   {
      if ( m_carried )
         m_carried->decref();
      m_carried = c;
      c->incref();
   }
};

template<class _T>
CoreObject* CoreCarrier_Factory( const CoreClass *cls, void *data, bool )
{
   return new CoreCarrier<_T>( cls, reinterpret_cast<_T*>(data));
}

}

#endif

/* end of corecarrier.h */
