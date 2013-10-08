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
#include <falcon/mt.h>
#include <falcon/wopi/sharedmem.h>
#include <falcon/pstep.h>

#include <map>

namespace Falcon {
namespace WOPI {

class VMContext;

/** WOPI engine.
   This class implements the WOPI host that is loaded by engine frames
   as web servers.

 */
class Wopi
{
public:
   Wopi();
   virtual ~Wopi();

   typedef enum {
        e_le_silent,
        e_le_log,
        e_le_kind,
        e_le_error
     }
     t_logMode;

     typedef enum {
        e_sm_file,
        e_sm_memory
     }
     t_sessionMode;

   bool setAppData( VMContext* ctx, Item& data, const String& appName, bool atomicUpdate );
   bool getAppData( VMContext* ctx, Item& data, const String& appName );

   bool setContextData( const String& id, const Item& data );
   bool getContextData( const String& id, Item& data ) const;

   void dataLocation( const String& loc );
   String dataLocation();

   void tempDir( const String& value ) { m_tmpDir = value; }
   const String& tempDir() const { return m_tmpDir; }

   void uploadDir( const String& value ) { m_uploadDir = value; }
   const String& uploadDir() const { return m_uploadDir; }

   void falconHandler( const String& value ) { m_falconHandler = value; }
   const String& falconHandler() const { return m_falconHandler; }

   void logMode( t_logMode value ) { m_logMode = value; }
   t_logMode logMode() const { return m_logMode; }

   void sessionMode( t_sessionMode value ) { m_sessionMode = value; }
   t_sessionMode logMode() const { return m_sessionMode; }

   void sessionMode( int32 value ) { m_maxUpload = value; }
   int32 logMode() const { return m_maxUpload; }

   void sessionTimeout( int32 value ) { m_sessionTimeout = value; }
   int32 sessionTimeout() const { return m_sessionTimeout; }

private:
   typedef std::map<String, SharedMem*> AppDataMap;
   /** Persistent data map.
      We have one of theese per thread.
   */
   typedef std::map<String, GCLock*> PDataMap;

   PStep* m_readAppDataNext;
   PStep* m_writeAppDataNext;

   Mutex m_mtx;
   String m_tmpDir;
   String m_uploadDir;
   String m_falconHandler;

   t_logMode m_logMode;
   t_sessionMode m_sessionMode;

   int32 m_maxUpload;
   int32 m_sessionTimeout;

   void inner_readAppData( SharedMem* shmem, Item& data );
   SharedMem* inner_create_appData( const String& appName );

   AppDataMap m_admap;

   String m_sAppDataLoc;

   /** Persistent data. */
   ThreadSpecific m_ctxdata;
   static void pdata_deletor( void* );

};


}
}

#endif /* _FALCON_WOPI_WOPI_H_ */

/* end of wopi.h */
