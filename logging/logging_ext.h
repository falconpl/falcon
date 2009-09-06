/*
   FALCON - The Falcon Programming Language.
   FILE: logging_ext.cpp

   Falcon VM interface to logging module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/



#ifndef FLC_LOGGING_EXT_H
#define FLC_LOGGING_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/coreobject.h>

#include <falcon/error.h>
#include <falcon/cclass.h>

#include <falcon/error_base.h>

#ifndef FALCON_LOGGING_ERROR_BASE
   #define FALCON_LOGGING_ERROR_BASE         1200
#endif
/*
#define FALCP_ERR_INVFORMAT  (FALCON_CONFPARSER_ERROR_BASE + 0)
#define FALCP_ERR_STORE      (FALCON_CONFPARSER_ERROR_BASE + 1)
*/
namespace Falcon {

//=====================================================
// CoreLogArea
//=====================================================

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


   virtual CoreObject *clone() const
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


namespace Ext {

// ==============================================
// Class LogArea
// ==============================================
FALCON_FUNC  LogArea_init( ::Falcon::VMachine *vm );
FALCON_FUNC  LogArea_add( ::Falcon::VMachine *vm );
FALCON_FUNC  LogArea_remove( ::Falcon::VMachine *vm );
FALCON_FUNC  LogArea_log( ::Falcon::VMachine *vm );

FALCON_FUNC  LogChannel_init( ::Falcon::VMachine *vm );
FALCON_FUNC  LogChannel_level( ::Falcon::VMachine *vm );
FALCON_FUNC  LogChannel_format( ::Falcon::VMachine *vm );

FALCON_FUNC  LogChannelStream_init( ::Falcon::VMachine *vm );

}
}

#endif

/* end of logging_ext.h */
