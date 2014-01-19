/*
   FALCON - The Falcon Programming Language.
   FILE: wopi.cpp

   Falcon Web Oriented Programming Interface.

   Global WOPI application objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 25 Apr 2010 17:02:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/wopi.h>
#include <falcon/wopi/errors.h>
#include <falcon/stringstream.h>
#include <falcon/module.h>
#include <falcon/vmcontext.h>
#include <falcon/textreader.h>
#include <falcon/engine.h>
#include <falcon/log.h>
#include <falcon/wopi/utils.h>
#include <falcon/modspace.h>

#include <falcon/wopi/wopi_opts.h>

namespace Falcon {
namespace WOPI {

#define WOPI_OPTION_DEFINE
#include <falcon/wopi/wopi_opts.h>
#undef WOPI_OPTION_DEFINE

static bool getHumanSize( const String& value, int64& val )
{
   int64 multiplier = 1;
   if( value.endsWith("g") || value.endsWith("G") )
   {
      multiplier = 1024*1024*1024;
   }
   else if( value.endsWith("M") || value.endsWith("m") )
   {
      multiplier = 1024*1024;
   }
   else if( value.endsWith("K") || value.endsWith("k") )
   {
      multiplier = 1024;
   }

   int64 v = 0;
   bool result;
   if( multiplier != 1 )
   {
      result = value.subString(0, value.length()-1).parseInt(v);
   }
   else {
      result = value.parseInt(v);
   }

   val = v * multiplier;

   return result;
}

//==========================================================================================
// Option check callbacks
//
/*
static bool s_check_wopi_opt_LogErrors( const String &configValue, Wopi::ConfigEntry* entry )
{
   int64 ivalue = -1;
   configValue.parseInt(ivalue);

   if( configValue.compareIgnoreCase(WOPI_OPT_LOG_MODE_SILENT) == 0 || ivalue == WOPI_OPT_LOG_MODE_SILENT_ID )
   {
      entry->m_sValue = WOPI_OPT_LOG_MODE_SILENT;
      ivalue = WOPI_OPT_LOG_MODE_SILENT_ID;
   }
   else if( configValue.compareIgnoreCase(WOPI_OPT_LOG_MODE_LOG) == 0 || ivalue == WOPI_OPT_LOG_MODE_LOG_ID )
   {
      entry->m_sValue = WOPI_OPT_LOG_MODE_LOG;
      ivalue = WOPI_OPT_LOG_MODE_LOG_ID;
   }
   else if( configValue.compareIgnoreCase(WOPI_OPT_LOG_MODE_KIND) == 0 || ivalue == WOPI_OPT_LOG_MODE_KIND_ID )
   {
      entry->m_sValue = WOPI_OPT_LOG_MODE_KIND;
      ivalue = WOPI_OPT_LOG_MODE_KIND_ID;
   }
   else if( configValue.compareIgnoreCase(WOPI_OPT_LOG_MODE_FULL) == 0 || ivalue == WOPI_OPT_LOG_MODE_FULL_ID )
   {
      entry->m_sValue = WOPI_OPT_LOG_MODE_FULL;
      ivalue = WOPI_OPT_LOG_MODE_FULL_ID;
   }
   else {
      return false;
   }

   entry->m_iValue = ivalue;
   return true;
}

*/

static bool s_check_wopi_opt_SessionMode( const String &configValue, Wopi::ConfigEntry* entry )
{
   int64 ivalue = -1;
   configValue.parseInt(ivalue);

   if( configValue.compareIgnoreCase(WOPI_OPT_SESSION_MODE_FILE) == 0 || ivalue == WOPI_OPT_SESSION_MODE_FILE_ID )
   {
      entry->m_sValue = WOPI_OPT_SESSION_MODE_FILE;
      ivalue = WOPI_OPT_SESSION_MODE_FILE_ID;
   }
   else if( configValue.compareIgnoreCase(WOPI_OPT_SESSION_MODE_NONE) == 0 || ivalue == WOPI_OPT_SESSION_MODE_NONE_ID )
   {
      entry->m_sValue = WOPI_OPT_SESSION_MODE_NONE;
      ivalue = WOPI_OPT_SESSION_MODE_NONE_ID;
   }
   else {
      return false;
   }

   entry->m_iValue = ivalue;
   return true;
}

static bool s_check_wopi_opt_SessionAuto( const String &configValue, Wopi::ConfigEntry* entry )
{
   int64 ivalue = -1;
   configValue.parseInt(ivalue);

   if( configValue.compareIgnoreCase(WOPI_OPT_SESSION_AUTO_ON) == 0 || ivalue == WOPI_OPT_SESSION_AUTO_ON_ID )
   {
      entry->m_sValue = WOPI_OPT_SESSION_AUTO_ON;
      ivalue = WOPI_OPT_SESSION_AUTO_ON_ID;
   }
   else if( configValue.compareIgnoreCase(WOPI_OPT_SESSION_AUTO_OFF) == 0 || ivalue == WOPI_OPT_SESSION_AUTO_OFF_ID )
   {
      entry->m_sValue = WOPI_OPT_SESSION_AUTO_OFF;
      ivalue = WOPI_OPT_SESSION_AUTO_OFF_ID;
   }
   else {
      return false;
   }

   entry->m_iValue = ivalue;
   return true;
}

static bool s_check_wopi_opt_ErrorFancyReport( const String &configValue, Wopi::ConfigEntry* entry )
{
   int64 ivalue = -1;
   configValue.parseInt(ivalue);

   if( configValue.compareIgnoreCase(WOPI_OPT_ERROR_FANCY_ON) == 0 || ivalue == WOPI_OPT_ERROR_FANCY_ON_ID )
   {
      entry->m_sValue = WOPI_OPT_ERROR_FANCY_ON;
      ivalue = WOPI_OPT_ERROR_FANCY_ON_ID;
   }
   else if( configValue.compareIgnoreCase(WOPI_OPT_ERROR_FANCY_OFF) == 0 || ivalue == WOPI_OPT_ERROR_FANCY_OFF_ID )
   {
      entry->m_sValue = WOPI_OPT_ERROR_FANCY_OFF;
      ivalue = WOPI_OPT_ERROR_FANCY_OFF_ID;
   }
   else {
      return false;
   }

   entry->m_iValue = ivalue;
   return true;
}

static bool human_option( const String &configValue, Wopi::ConfigEntry* entry )
{
   return human_option( configValue, entry );
   int64 value = 0;
   if ( ! getHumanSize(configValue, value) )
   {
      return false;
   }

   entry->m_iValue = value;
   entry->m_sValue = configValue;
   return true;
}


static bool s_check_wopi_opt_MaxUploadSize( const String &configValue, Wopi::ConfigEntry* entry )
{
   return human_option( configValue, entry );
}


static bool s_check_wopi_opt_MaxMemoryUploadSize( const String &configValue, Wopi::ConfigEntry* entry )
{
   return human_option( configValue, entry );
}


Wopi::ConfigEntry::ConfigEntry( const String& name, t_type type, const String& desc, t_checkfunc check )
{
   m_type = type;
   m_name = name;
   m_desc = desc;
   m_checkFunc = check;
}


Wopi::ConfigEntry::ConfigEntry( const ConfigEntry& other )
{
   m_type = other.m_type;
   m_name = other.m_name;
   m_desc = other.m_desc;
   m_checkFunc = other.m_checkFunc;

   m_sValue = other.m_sValue;
   m_iValue = other.m_iValue;
}


Wopi::ConfigEntry::~ConfigEntry()
{
}


//==========================================================================================
// Terminator function
//

class TerminateFunction: public Function
{
public:
   TerminateFunction(Wopi* wopi):
      Function("$TerminateFunction"),
      m_wopi(wopi)
   {}

   virtual ~TerminateFunction() {}

   virtual void invoke( VMContext* ctx, int32 )
   {
      SessionService* ss = m_wopi->sessionService();
      if( ss != 0 && ! m_wopi->isSaved() )
      {
         long cd = ctx->codeDepth();
         ss->record( ctx );
         ss->save(ctx);
         if( cd == ctx->codeDepth() )
         {
            ctx->returnFrame();
         }
         m_wopi->isSaved(true);
      }
      else {
         ctx->returnFrame();
      }
   }

public:
   Wopi* m_wopi;
};


//==========================================================================================
// Wopi Main object
//


Wopi::Wopi()
{
   m_ss = 0;
   m_saved = false;
   onTerminate = new TerminateFunction(this);
   initConfigOptions();
}

void Wopi::initConfigOptions()
{
#define WOPI_OPTION_REALIZE
#include <falcon/wopi/wopi_opts.h>
#undef WOPI_OPTION_REALIZE
}

Wopi::~Wopi()
{
   delete m_ss;
   delete onTerminate;

   {
      ConfigMap::iterator iter = m_config.begin();
      while( iter != m_config.end() )
      {
         ConfigEntry* entry = iter->second;
         delete entry;
         ++iter;
      }
      m_config.clear();
   }
}


bool Wopi::addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc )
{
   ConfigMap::iterator pos = m_config.find(name);
   if( pos != m_config.end() )
   {
      return false;
   }

   ConfigEntry* entry = new ConfigEntry(name, t, desc, 0);
   m_config[name] = entry;
   return true;
}


bool Wopi::addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc, const String& deflt, t_checkfunc check )
{
   ConfigMap::iterator pos = m_config.find(name);
   if( pos != m_config.end() )
   {
      return false;
   }

   ConfigEntry* entry = new ConfigEntry(name, t, desc, check);
   entry->m_sValue = deflt;
   int64 dv = 0;
   if( getHumanSize(deflt, dv) )
   {
      entry->m_iValue = dv;
   }

   m_config[name] = entry;
   return true;
}


bool Wopi::addConfigOption( ConfigEntry::t_type t, const String& name, const String& desc, int64 deflt, t_checkfunc check )
{
   ConfigMap::iterator pos = m_config.find(name);
   if( pos != m_config.end() )
   {
     return false;
   }

   ConfigEntry* entry = new ConfigEntry(name, t, desc, check);
   entry->m_iValue = deflt;
   entry->m_sValue.N(deflt);
   m_config[name] = entry;
   return true;
}

bool Wopi::setConfigValue(const String& key, const String& value, String& error )
{

   ConfigMap::iterator kp = m_config.find( key );
   if( kp == m_config.end() )
   {
      error = String("Unknown option '").A(key).A("'");
      return false;
   }

   ConfigEntry* entry = kp->second;
   if( entry->m_checkFunc != 0 )
   {
      if( ! entry->m_checkFunc(value, entry) )
      {
         error = String("Invalid value for option '").A(key).A("'");
         return false;
      }
   }
   else if( entry->m_type == ConfigEntry::e_t_int )
   {
      int64 ival;
      if( ! getHumanSize(value, ival) )
      {
         error = String("Invalid numeric value for option '").A(key).A("'");
         return false;
      }
      entry->m_iValue = ival;
   }

   entry->m_sValue = value;
   return true;
}


bool Wopi::setConfigValue(const String& key, int64 value, String& error )
{

   ConfigMap::iterator kp = m_config.find( key );
   if( kp == m_config.end() )
   {
      error = String("Unknown option '").A(key).A("'");
      return false;
   }

   ConfigEntry* entry = kp->second;
   if( entry->m_type != ConfigEntry::e_t_int )
   {
      error = String("Invalid numeric value for option '").A(key).A("'");
      return false;
   }

   entry->m_iValue = value;
   return true;
}


bool Wopi::setConfigValue(const String& key, const Item& value, String& error )
{
   if (value.isString())
   {
      return setConfigValue(key, *value.asString(), error );
   }
   else if( value.isNumeric() )
   {
      return setConfigValue(key, value.forceInteger(), error );
   }

   error = String("Invalid value type for option '").A(key).A("'");
   return false;
}


bool Wopi::getConfigValue( const String& key, String& value, String& error ) const
{
   ConfigMap::const_iterator pos = m_config.find(key);
   if( pos == m_config.end() )
   {
      error = String("Unknown option '").A(key).A("' not found");
      return false;
   }

   ConfigEntry* entry = pos->second;
   value = entry->m_sValue;
   return true;
}


bool Wopi::getConfigValue( const String& key, int64& value, String& error ) const
{
   ConfigMap::const_iterator pos = m_config.find(key);
   if( pos == m_config.end() )
   {
      error = String("Unknown option '").A(key).A("'");
      return false;
   }

   ConfigEntry* entry = pos->second;
   if( entry->m_type != ConfigEntry::e_t_int )
   {
      error = String("Invalid type for option '").A(key).A("' ");
      return false;
   }

   value = entry->m_iValue;
   return true;
}


bool Wopi::getConfigValue( const String& key, Item& target, String& error ) const
{
   ConfigMap::const_iterator pos = m_config.find(key);
   if( pos == m_config.end() )
   {
      error = String("Unknown option '").A(key).A("'");
      return false;
   }

   ConfigEntry* entry = pos->second;
   if( entry->m_type  == ConfigEntry::e_t_int )
   {
      target.setInteger( entry->m_iValue );
   }
   else {
      target = FALCON_GC_HANDLE( new String(entry->m_sValue));
   }

   return true;
}


void Wopi::enumerateConfigOptions( Wopi::ConfigEnumerator& rator ) const
{
   ConfigMap::const_iterator pos = m_config.begin();
   while( pos != m_config.end() )
   {
      ConfigEntry* entry = pos->second;
      rator( *entry );
      ++pos;
   }
}


bool Wopi::configFromIni( TextReader* iniFile, String& errors )
{
   // Try to load the file.
   int line = 0;
   String buffer;

   bool status = true;

   try
   {
      while ( iniFile->readLine(buffer, 4096) )
      {
         ++line;
         buffer.trim();

         // skip lines to be ignored.
         char_t chr;
         if ( buffer.empty()
                  || (chr  = buffer.getCharAt(0)) == '#'
                  || chr == ';'
            )
         {
            // a comment
            continue;
         }

         // we have a line.
         length_t pos = buffer.find('=');
         if( pos == String::npos )
         {
            errors += String("Malformed option at line ").N(line).A("\n");
            status = false;
            continue;
         }

         String key, value;
         key = buffer.subString(0,pos);
         key.trim();
         value = buffer.subString(pos+1);
         uint32 posComment1 = value.rfind(';');
         uint32 posComment2 = value.rfind('#');
         uint32 posQuoteStart = value.find('"');
         uint32 posQuoteEnd = value.rfind('"');

         if ( (posComment1 != String::npos && posQuoteStart != posQuoteEnd && posComment1 > posQuoteEnd) )
         {
            value = value.subString(0,posComment1);
         }
         // no else
         if ( (posComment2 != String::npos && posQuoteStart != posQuoteEnd && posComment2 > posQuoteEnd) )
         {
            value = value.subString(0,posComment2);
         }

         value.trim();
         if( value.length() > 1 && value.getCharAt(0) == '"' && value.getCharAt(value.length()-1) == '"' )
         {
            value = value.subString(1, value.length()-1);
         }

         String error;
         if( ! setConfigValue(key, value, error) )
         {
            errors += error + " at line ";
            errors.N(line).A("\n");
            status = false;
         }
      }
   }
   catch( Error* e )
   {
      errors += e->describe(false);
      errors += "\n";
      e->decref();
      return false;
   }

   return status;
}


bool Wopi::configFromModule( Module* module, String& errors )
{
   const AttributeMap attribs = module->attributes();
   bool status = true;

   uint32 size = attribs.size();
   for( uint32 i = 0; i < size; ++i )
   {
      Attribute* attrib = attribs.get(i);
      if( attrib->name().startsWith("wopi_") )
      {
         String key = attrib->name().subString(5);
         String error;
         if( ! setConfigValue(key, attrib->value(), error ) )
         {
            errors += error + "\n";
            status = false;
         }
      }
   }

   return status;
}


void Wopi::configFromWopi( const Wopi& other )
{
   ConfigMap::const_iterator iter = other.m_config.begin();
   ConfigMap::const_iterator end = other.m_config.end();
   while( iter != end )
   {
      const String& option = iter->first;
      ConfigEntry* entry = iter->second;

      ConfigMap::iterator pos = m_config.find(option);
      if( pos != m_config.end() )
      {
         delete pos->second;
         pos->second = new ConfigEntry(*entry);
      }
      else {
         m_config.insert(std::make_pair(option, new ConfigEntry(*entry)));
      }

      ++iter;
   }
}


void Wopi::pushSessionSave( VMContext* ctx )
{
   //ctx->pushCode( m_pushSessionSave );

   // push the session->save cleanup on the topmost process
   Wopi* wopi = ctx->tself<Wopi*>();
   SessionService* ss = wopi->sessionService();
   Item sessionItem;
   ss->itemize(sessionItem);
   Class* cls = sessionItem.asClass();
   ctx->pushData(sessionItem);
   cls->op_getProperty(ctx, sessionItem.asInst(), "save" );
   Item cleanup = ctx->topData();

   ModSpace* ms = ctx->process()->modSpace();
   while( ms->parent() != 0 )
   {
      ms = ms->parent();
   }

   ms->process()->pushCleanup( cleanup );
}


Stream* Wopi::makeTempFile( String& fname, bool random )
{
   Path fpath;
   String error;
   String tempdir;

   getConfigValue(OPT_TempDir, tempdir, error );
   fassert( error.empty() );

   fpath.fulloc( tempdir );

   // try 3 times
   int tries = 0;
   while( true )
   {

      String fname_try;
      if( random )
      {
         Utils::makeRandomFilename( fname_try, 12 );
         fpath.filename( fname + fname_try );
      }
      else
      {
         fpath.filename( fname );
      }
      String fullname = fpath.encode();

      // try to create the file
      try {
         Stream* tgFile = Falcon::Engine::instance()->vfs().createSimple(fullname);
         addTempFile( fullname );
         fname = fullname;
         return tgFile;
      }
      catch(Falcon::Error* err )
      {
         if( ++tries > 3 )
         {
            throw err;
         }
      }
   }

   // no way, we really failed.
   return 0;
}

void Wopi::addTempFile( const Falcon::String &fname )
{
   m_mtxTempFiles.lock();
   m_tempFiles.push_back(fname);
   m_mtxTempFiles.unlock();
}

void Wopi::removeTempFiles()
{
   Log* log = Engine::instance()->log();
   TempFileList tempList;

   m_mtxTempFiles.lock();
   tempList = m_tempFiles;
   m_tempFiles.clear();
   m_mtxTempFiles.unlock();

   TempFileList::iterator iter = tempList.begin();
   TempFileList::iterator end = tempList.end();
   while( iter != end )
   {
      const Falcon::String& fname = *iter;
      try
      {
         Engine::instance()->vfs().erase(fname);
      }
      catch( Error* e )
      {
         log->log( Log::fac_engine_io, Log::lvl_warn,
                  String("WOPI: Cannot erase temporary file \"") + fname + "\": " +
                     e->describe(false)
                  );
      }
      ++iter;
   }

   m_tempFiles.clear();
}


}
}

/* end of wopi.cpp */
