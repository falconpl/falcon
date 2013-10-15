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

#include <map>

namespace Falcon {
class TextReader;

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
      ~ConfigEntry();
   };


   bool setConfigValue( const String& key, const String& value, String& error );
   bool setConfigValue( const String& key, int64 value, String& error );
   bool setConfigValue( const String& key, const Item& value, String& error );

   bool getConfigValue( const String& key, String& value, String& error ) const;
   bool getConfigValue( const String& key, int64& value, String& error ) const;
   bool getConfigValue( const String& key, Item& target, String& error ) const;


   bool addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc );
   bool addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc, String& deflt, t_checkfunc check = 0 );
   bool addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc, int64 deflt, t_checkfunc check = 0 );

   class ConfigEnumerator
   {
   public:
      inline virtual ~ConfigEnumerator() {}
      virtual void operator() (ConfigEntry& entry) = 0;
   };

   void enumerateConfigOptions( ConfigEnumerator& rator );

   //============================================================
   // By-context data
   //
   bool setContextData( const String& id, const Item& data );
   bool getContextData( const String& id, Item& data ) const;
   bool removeContextData( const String& id );

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
    * */
   Stream* makeTempFile( String& fname );

   /** Adds a temporary file.
      The VM tries to delete all the temporary files during its destructor.
      On failure, it ingores the problem and logs an error in Apache.
   */
   void addTempFile( const Falcon::String &fname );

   /** Gets the list of temporary files.
      Before the VM is destroyed, this should be taken out
      so that it is then possible to get rid of the files.

      Using removeTempFiles after the VM has been destroyed ensures
      that all the streams open by the VM are closed (as this is done
      during the GC step).

      \return an opaque pointer to an internal structure.
   */
   void* getTempFiles() const { return m_tempFiles; }

   /** Removes from the disk a list of temporary files.

      Using removeTempFiles after the VM has been destroyed ensures
      that all the streams open by the VM are closed (as this is done
      during the GC step).

      \param head The valued returned from getTempFiles() before the VM was destroyed.
      \param data Opaque pointer passed as extra data to the error_func (can be 0 if not used).
      \param error_func callback that will be invoked in some file can't be deleted.
   */
   static void removeTempFiles( void* head, void* data, void (*error_func)(const String& msg, void* data) );

private:
   /** Persistent data map.
      We have one of these per thread.
   */
   typedef std::map<String, GCLock*> PDataMap;
   typedef std::map<String, ConfigEntry*> ConfigMap;

   ConfigMap m_config;

   PStep* m_readAppDataNext;
   PStep* m_writeAppDataNext;

   /** Persistent data. */
   ThreadSpecific m_ctxdata;
   static void pdata_deletor( void* );

   void initConfigOptions();
};


}
}

#endif /* _FALCON_WOPI_WOPI_H_ */

/* end of wopi.h */
