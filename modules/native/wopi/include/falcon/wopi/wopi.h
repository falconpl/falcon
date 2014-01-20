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
#include <falcon/pstep.h>
#include <falcon/gclock.h>

#include <falcon/wopi/apploglistener.h>
#include <falcon/wopi/webloglistener.h>

#include "../../feathers/shmem/session_srv.h"

#include <map>
#include <list>

namespace Falcon {
class TextReader;
class Stream;
class VMContext;

namespace WOPI {

#define WOPI_OPTION_DECLARE
#include <falcon/wopi/wopi_opts.h>
#undef WOPI_OPTION_DECLARE


/** WOPI engine.
   This class implements the WOPI host that is loaded by engine frames
   as web servers.

 */
class Wopi
{
public:
   Wopi();
   virtual ~Wopi();

   /*
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
   */

   //============================================================
   // Configuration management
   //

   class ConfigEntry;
   typedef bool (*t_checkfunc)(const String &configValue, ConfigEntry* entry );

   /** Class representing a single configuration entry.
    *
    */
   class ConfigEntry
   {
   public:

      typedef enum {
        e_t_int,
        e_t_string
      }
      t_type;


      t_type m_type;
      String m_sValue;
      int64 m_iValue;
      String m_name;
      String m_desc;
      t_checkfunc m_checkFunc;

      ConfigEntry( const String& name, t_type type, const String& desc, t_checkfunc check = 0 );
      ConfigEntry( const ConfigEntry& other );
      ~ConfigEntry();
   };


   bool setConfigValue( const String& key, const String& value, String& error );
   bool setConfigValue( const String& key, int64 value, String& error );
   bool setConfigValue( const String& key, const Item& value, String& error );

   bool getConfigValue( const String& key, String& value, String& error ) const;
   bool getConfigValue( const String& key, int64& value, String& error ) const;
   bool getConfigValue( const String& key, Item& target, String& error ) const;

   bool addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc );
   bool addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc, const String& deflt, t_checkfunc check = 0 );
   bool addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc, int64 deflt, t_checkfunc check = 0 );

   class ConfigEnumerator
   {
   public:
      inline virtual ~ConfigEnumerator() {}
      virtual void operator() (ConfigEntry& entry) = 0;
   };

   void enumerateConfigOptions( ConfigEnumerator& rator ) const;

   //============================================================
   // High level configs
   //
   bool configFromIni( TextReader* iniFile, String& errors );
   bool configFromModule( Module* module, String& errors );

   //============================================================
   // Temporary file management
   //

   /** Create a generic usage temporary file.
    \throw IoError on error.
    \param fname A file name for the temporary file.
    \param bRandom if true, add a random suffix to the filename.
    \return A stream open for writing.

    On exit, fname will hold the full path to the created temporary file.

    The temporary file will be removed when the wopi object is destroyed.
    */
   Stream* makeTempFile( String& fname, bool bRandom );

   /**
    * Creates an anonymous temporary file
    * \return A stream open for writing.
    * \throw IoError on error.
    */
   Stream* makeTempFile() { String temp = ""; return makeTempFile( temp, true); }

   /** Adds a temporary file.
      The VM tries to delete all the temporary files during its destructor.
      On failure, it ignores the problem and logs an error to the log system.
   */
   void addTempFile( const Falcon::String &fname );

   /** Removes from the disk a list of temporary files.
    \note This method is threadsafe, and can be called multiple
       times during the execution of the process where the wopi module
       is loaded (although it should be called when it's safe to
       assume that the temporary files are not needed anymore).
   */
   void removeTempFiles();

   /** Copies the configuration from a template WOPI object */
   void configFromWopi( const Wopi& other );

   SessionService* sessionService() const { return m_ss; }
   void sessionService( SessionService* ss ) { m_ss = ss; }

   void pushSessionSave( VMContext* ctd );

   // Function injected in main modules to be invoked at termination.
   Function* onTerminate;

   bool isSaved() const { return m_saved; }
   void isSaved( bool b ) { m_saved = b; }

   /** Invoked by the script runner before starting a script */
   void setupLogListener();

   /** Utility setting the application listener level as required by the config values. */
   void configureAppLogListener( Log::Listener* ll );

   void renderWebLogs( Stream* target );
   void removeLogListener();

private:
   typedef std::map<String, ConfigEntry*> ConfigMap;

   typedef std::list<String> TempFileList;

   Mutex m_mtxTempFiles;
   TempFileList m_tempFiles;

   ConfigMap m_config;

   PStep* m_readAppDataNext;
   PStep* m_writeAppDataNext;

   SessionService* m_ss;
   bool m_saved;
   WebLogListener* m_webll;

   void initConfigOptions();
};


}
}

#endif /* _FALCON_WOPI_WOPI_H_ */

/* end of wopi.h */
