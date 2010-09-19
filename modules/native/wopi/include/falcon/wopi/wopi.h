/*
   FALCON - The Falcon Programming Language.
   FILE: wopi.h

   Falcon Web Oriented Programming Interface.

   Global WOPI application objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 25 Apr 2010 17:02:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_WOPI_H_
#define _FALCON_WOPI_WOPI_H_

#include <falcon/itemdict.h>
#include <falcon/string.h>
#include <falcon/cclass.h>
#include <falcon/coreobject.h>
#include <falcon/mt.h>
#include <falcon/wopi/sharedmem.h>

#define FALCON_WOPI_PDATADIR_ATTRIB "wopi_pdataDir"

#include <map>

namespace Falcon {
namespace WOPI {

class Wopi: public BaseAlloc
{
public:
   Wopi();
   virtual ~Wopi();

   bool setData( Item& data, const String& appName, bool atomicUpdate );
   bool getData( Item& data, const String& appName );

   bool setPersistent( const String& id, const Item& data );
   bool getPeristent( const String& id, Item& data ) const;

   void dataLocation( const String& loc );
   String dataLocation();

private:
   void inner_readData( SharedMem* shmem, Item& data );
   SharedMem* inner_create_appData( const String& appName );

   Mutex m_mtx;
   typedef std::map<String, SharedMem*> AppDataMap;
   /** Persistent data map.
      We have one of theese per thread.
   */
   typedef std::map<String, GarbageLock*> PDataMap;
   AppDataMap m_admap;

   String m_sAppDataLoc;

   /** Persistent data. */
   ThreadSpecific m_pdata;
   static void pdata_deletor( void* );
};


/** Main WOPI object.
    Central object of the WOPI system.
*/
class CoreWopi: public CoreObject
{
public:
   CoreWopi( const CoreClass* parent );
   virtual ~CoreWopi();

   virtual CoreObject *clone() const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &prop, Item &value ) const;
   void configFromModule( const Module* mod );

   static CoreObject* factory( const CoreClass *cls, void *user_data, bool bDeserializing );


   Wopi* wopi() const { return m_wopi; }
   void setWopi( Wopi* w ) { m_wopi = w; }
private:

   Wopi* m_wopi;
};

}
}

#endif /* _FALCON_WOPI_WOPI_H_ */

/* end of wopi.h */
